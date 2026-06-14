from typing import Optional, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .config import MoEConfig, BalancerConfig, DistributedConfig
from .utils import all_reduce_sum, all_reduce_mean, get_logger

logger = get_logger(__name__)


@dataclass
class UtilizationStats:
    step_utilization: torch.Tensor
    moving_avg_utilization: torch.Tensor
    cumulative_utilization: torch.Tensor
    total_tokens_processed: int
    low_utilization_experts: torch.Tensor
    high_utilization_experts: torch.Tensor
    utilization_variance: float
    utilization_cv: float
    balance_ratio: float


class MovingAverageUtilizationTracker:
    def __init__(
        self,
        moe_config: MoEConfig,
        balancer_config: BalancerConfig,
        dist_config: Optional[DistributedConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.moe_config = moe_config
        self.balancer_config = balancer_config
        self.dist_config = dist_config or DistributedConfig()
        self.device = device or torch.device("cpu")

        self.num_experts = moe_config.num_experts
        self.momentum = balancer_config.utilization_momentum
        self.warmup_steps = balancer_config.utilization_warmup_steps
        self.threshold_low = balancer_config.utilization_threshold_low
        self.threshold_high = balancer_config.utilization_threshold_high

        self.step_count = 0
        self.register_buffers()

    def register_buffers(self):
        self._step_utilization = torch.zeros(self.num_experts, device=self.device, dtype=torch.float32)
        self._moving_avg_utilization = torch.zeros(self.num_experts, device=self.device, dtype=torch.float32)
        self._cumulative_utilization = torch.zeros(self.num_experts, device=self.device, dtype=torch.float64)
        self._total_tokens_processed = torch.tensor(0, device=self.device, dtype=torch.int64)
        self._bias_correction_term = torch.tensor(1.0, device=self.device, dtype=torch.float32)

    def to(self, device: torch.device):
        self.device = device
        self._step_utilization = self._step_utilization.to(device)
        self._moving_avg_utilization = self._moving_avg_utilization.to(device)
        self._cumulative_utilization = self._cumulative_utilization.to(device)
        self._total_tokens_processed = self._total_tokens_processed.to(device)
        self._bias_correction_term = self._bias_correction_term.to(device)
        return self

    def update(
        self,
        expert_weights: torch.Tensor,
        expert_indices: Optional[torch.Tensor] = None,
    ) -> None:
        num_tokens = expert_weights.shape[0]

        if expert_indices is not None:
            top_k = expert_indices.shape[-1]
            flat_indices = expert_indices.view(-1)
            flat_weights = expert_weights.view(-1)
            step_util = torch.zeros(self.num_experts, device=self.device, dtype=torch.float32)
            step_util.index_add_(0, flat_indices, flat_weights.float())
        else:
            step_util = expert_weights.sum(dim=0).float()

        step_util = step_util / max(num_tokens, 1)

        if self.dist_config.use_distributed and dist.is_initialized():
            step_util = all_reduce_mean(step_util)
            num_tokens_tensor = torch.tensor(num_tokens, device=self.device, dtype=torch.int64)
            num_tokens_tensor = all_reduce_sum(num_tokens_tensor)
            num_tokens = num_tokens_tensor.item()

        self._step_utilization.copy_(step_util)
        self._cumulative_utilization.add_(step_util.double())
        self._total_tokens_processed.add_(num_tokens)

        self.step_count += 1

        if self.step_count <= self.warmup_steps:
            alpha = 1.0 / max(self.step_count, 1)
            self._moving_avg_utilization.mul_(1 - alpha).add_(step_util, alpha=alpha)
        else:
            self._moving_avg_utilization.mul_(self.momentum).add_(step_util, alpha=1 - self.momentum)

        self._bias_correction_term.mul_(self.momentum).add_(1 - self.momentum)

    def get_bias_corrected_utilization(self) -> torch.Tensor:
        bc = self._bias_correction_term.item()
        if bc > 0 and self.step_count > self.warmup_steps:
            return self._moving_avg_utilization / bc
        return self._moving_avg_utilization.clone()

    def get_stats(self) -> UtilizationStats:
        bc_util = self.get_bias_corrected_utilization()
        mean_util = bc_util.mean().clamp_min(1e-9).item()
        std_util = bc_util.std().item()

        low_mask = bc_util < (mean_util * self.threshold_low)
        high_mask = bc_util > (mean_util * self.threshold_high)

        cv = std_util / mean_util
        ratio = (bc_util.max() / bc_util.clamp_min(1e-9).min()).item()

        return UtilizationStats(
            step_utilization=self._step_utilization.clone(),
            moving_avg_utilization=bc_util.clone(),
            cumulative_utilization=self._cumulative_utilization.clone(),
            total_tokens_processed=self._total_tokens_processed.item(),
            low_utilization_experts=torch.nonzero(low_mask, as_tuple=False).squeeze(-1),
            high_utilization_experts=torch.nonzero(high_mask, as_tuple=False).squeeze(-1),
            utilization_variance=std_util ** 2,
            utilization_cv=cv,
            balance_ratio=ratio,
        )

    def get_underutilized_experts(
        self,
        threshold_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        threshold = threshold_ratio or self.threshold_low
        bc_util = self.get_bias_corrected_utilization()
        mean_util = bc_util.mean().clamp_min(1e-9)
        mask = bc_util < (mean_util * threshold)
        return torch.nonzero(mask, as_tuple=False).squeeze(-1)

    def get_overutilized_experts(
        self,
        threshold_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        threshold = threshold_ratio or self.threshold_high
        bc_util = self.get_bias_corrected_utilization()
        mean_util = bc_util.mean().clamp_min(1e-9)
        mask = bc_util > (mean_util * threshold)
        return torch.nonzero(mask, as_tuple=False).squeeze(-1)

    def get_utilization_priority_scores(self) -> torch.Tensor:
        bc_util = self.get_bias_corrected_utilization()
        mean_util = bc_util.mean().clamp_min(1e-9)
        ratio = mean_util / (bc_util + 1e-9)
        scores = torch.log(ratio.clamp_min(1e-3))
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores

    def compute_load_imbalance_penalty(self) -> torch.Tensor:
        bc_util = self.get_bias_corrected_utilization()
        n = self.num_experts
        ideal = 1.0 / n
        penalty = n * ((bc_util - ideal) ** 2).sum()
        return penalty

    def reset(self):
        self.step_count = 0
        self.register_buffers()

    def state_dict(self) -> Dict:
        return {
            "step_count": self.step_count,
            "step_utilization": self._step_utilization.clone().cpu(),
            "moving_avg_utilization": self._moving_avg_utilization.clone().cpu(),
            "cumulative_utilization": self._cumulative_utilization.clone().cpu(),
            "total_tokens_processed": self._total_tokens_processed.clone().cpu(),
            "bias_correction_term": self._bias_correction_term.clone().cpu(),
        }

    def load_state_dict(self, state: Dict, map_location: Optional[torch.device] = None):
        dev = map_location or self.device
        self.step_count = state["step_count"]
        self._step_utilization.copy_(state["step_utilization"].to(dev))
        self._moving_avg_utilization.copy_(state["moving_avg_utilization"].to(dev))
        self._cumulative_utilization.copy_(state["cumulative_utilization"].to(dev))
        self._total_tokens_processed.copy_(state["total_tokens_processed"].to(dev))
        self._bias_correction_term.copy_(state["bias_correction_term"].to(dev))
