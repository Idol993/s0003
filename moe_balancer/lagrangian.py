from typing import Optional, Tuple, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BalancerConfig
from .utils import get_logger

logger = get_logger(__name__)


@dataclass
class LagrangianState:
    lambda_plus: torch.Tensor
    lambda_minus: torch.Tensor
    momentum_plus: torch.Tensor
    momentum_minus: torch.Tensor
    constraint_violation_plus: torch.Tensor
    constraint_violation_minus: torch.Tensor
    aug_lagrangian_value: float
    penalty_term: float


class LagrangianBalancer(nn.Module):
    def __init__(
        self,
        num_experts: int,
        balancer_config: BalancerConfig,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.config = balancer_config
        self.device = device or torch.device("cpu")

        init_lambda = balancer_config.lagrangian_init_lambda
        max_lambda = balancer_config.lagrangian_max_lambda
        min_lambda = balancer_config.lagrangian_min_lambda
        self.register_buffer("_max_lambda", torch.tensor(max_lambda, device=self.device))
        self.register_buffer("_min_lambda", torch.tensor(min_lambda, device=self.device))

        lambda_plus = torch.full((num_experts,), init_lambda, device=self.device, dtype=torch.float32)
        lambda_minus = torch.full((num_experts,), init_lambda, device=self.device, dtype=torch.float32)
        self.register_buffer("_lambda_plus", lambda_plus)
        self.register_buffer("_lambda_minus", lambda_minus)

        momentum_plus = torch.zeros(num_experts, device=self.device, dtype=torch.float32)
        momentum_minus = torch.zeros(num_experts, device=self.device, dtype=torch.float32)
        self.register_buffer("_momentum_plus", momentum_plus)
        self.register_buffer("_momentum_minus", momentum_minus)

        self._step_count = 0
        self._last_violation_plus = torch.zeros(num_experts, device=self.device)
        self._last_violation_minus = torch.zeros(num_experts, device=self.device)

    def to(self, device: torch.device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)

    def _get_target_utilization(self) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.num_experts
        target = 1.0 / n
        margin_low = target * (1.0 - self.config.utilization_threshold_low)
        margin_high = target * (1.0 + self.config.utilization_threshold_high - 1.0)
        upper_bound = target + margin_high
        lower_bound = max(target - margin_low, 0.0)
        return (
            torch.full((n,), lower_bound, device=self.device),
            torch.full((n,), upper_bound, device=self.device),
        )

    def compute_constraint_violations(
        self,
        expert_utilization: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lower_bounds, upper_bounds = self._get_target_utilization()
        violation_plus = F.relu(lower_bounds - expert_utilization)
        violation_minus = F.relu(expert_utilization - upper_bounds)
        return violation_plus, violation_minus

    def compute_load_balance_penalty(
        self,
        gate_logits: torch.Tensor,
        expert_weights: Optional[torch.Tensor] = None,
        expert_utilization: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        num_tokens = gate_logits.shape[0]
        n = self.num_experts

        if expert_utilization is None:
            if expert_weights is None:
                expert_weights = F.softmax(gate_logits, dim=-1)
            expert_utilization = expert_weights.sum(dim=0) / num_tokens

        expert_utilization = expert_utilization.to(self.device)

        violation_plus, violation_minus = self.compute_constraint_violations(expert_utilization)

        self._last_violation_plus.copy_(violation_plus)
        self._last_violation_minus.copy_(violation_minus)

        lambda_plus = self._lambda_plus.clone()
        lambda_minus = self._lambda_minus.clone()

        linear_term_plus = (lambda_plus * violation_plus).sum()
        linear_term_minus = (lambda_minus * violation_minus).sum()

        mu = self.config.lagrangian_lr * 10.0
        quadratic_term = 0.5 * mu * ((violation_plus ** 2).sum() + (violation_minus ** 2).sum())

        aug_lagrangian = linear_term_plus + linear_term_minus + quadratic_term

        cv_sq_penalty = n * ((expert_utilization - 1.0 / n) ** 2).sum()

        log_probs = F.log_softmax(gate_logits, dim=-1)
        probs = F.softmax(gate_logits, dim=-1)
        f_i = probs.mean(dim=0)
        log_f_i = torch.log(f_i.clamp_min(1e-9))
        load_loss = (f_i * log_f_i).sum() * n

        total_penalty = (
            aug_lagrangian
            + self.config.load_balance_loss_weight * cv_sq_penalty
            + self.config.aux_loss_weight * load_loss
        )

        info = {
            "aug_lagrangian": aug_lagrangian.item(),
            "cv_sq_penalty": cv_sq_penalty.item(),
            "load_loss": load_loss.item(),
            "violation_plus_sum": violation_plus.sum().item(),
            "violation_minus_sum": violation_minus.sum().item(),
            "lambda_plus_mean": lambda_plus.mean().item(),
            "lambda_minus_mean": lambda_minus.mean().item(),
            "total_penalty": total_penalty.item(),
        }

        return total_penalty, info

    def compute_implicit_gradient_bias_correction(
        self,
        gate_weights: torch.Tensor,
        force_assign_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if force_assign_mask is None or not force_assign_mask.any():
            return torch.tensor(0.0, device=self.device)

        num_tokens = gate_weights.shape[0]
        normal_tokens = ~force_assign_mask.any(dim=-1)
        forced_tokens = force_assign_mask.any(dim=-1)

        if not normal_tokens.any() or not forced_tokens.any():
            return torch.tensor(0.0, device=self.device)

        n_forced = forced_tokens.sum().float()
        n_normal = normal_tokens.sum().float()
        correction_ratio = self.config.bias_correction_factor

        forced_weights = gate_weights[forced_tokens]
        normal_weights = gate_weights[normal_tokens]

        forced_util = forced_weights.sum(dim=0) / n_forced.clamp_min(1.0)
        normal_util = normal_weights.sum(dim=0) / n_normal.clamp_min(1.0)

        bias = (forced_util - normal_util).abs().mean()

        correction = correction_ratio * bias
        return correction

    def step_multipliers(
        self,
        expert_utilization: Optional[torch.Tensor] = None,
    ) -> Dict:
        self._step_count += 1

        if expert_utilization is None:
            violation_plus = self._last_violation_plus
            violation_minus = self._last_violation_minus
        else:
            expert_utilization = expert_utilization.to(self.device)
            violation_plus, violation_minus = self.compute_constraint_violations(expert_utilization)
            self._last_violation_plus.copy_(violation_plus)
            self._last_violation_minus.copy_(violation_minus)

        lr = self.config.lagrangian_lr
        momentum = self.config.lagrangian_momentum
        wd = self.config.lagrangian_weight_decay

        self._momentum_plus.mul_(momentum).add_(violation_plus, alpha=1 - momentum)
        self._momentum_minus.mul_(momentum).add_(violation_minus, alpha=1 - momentum)

        if wd > 0:
            self._lambda_plus.mul_(1 - wd * lr)
            self._lambda_minus.mul_(1 - wd * lr)

        self._lambda_plus.add_(self._momentum_plus, alpha=lr)
        self._lambda_minus.add_(self._momentum_minus, alpha=lr)

        self._lambda_plus.clamp_(self._min_lambda.item(), self._max_lambda.item())
        self._lambda_minus.clamp_(self._min_lambda.item(), self._max_lambda.item())

        return {
            "step": self._step_count,
            "lambda_plus_mean": self._lambda_plus.mean().item(),
            "lambda_plus_max": self._lambda_plus.max().item(),
            "lambda_plus_min": self._lambda_plus.min().item(),
            "lambda_minus_mean": self._lambda_minus.mean().item(),
            "lambda_minus_max": self._lambda_minus.max().item(),
            "lambda_minus_min": self._lambda_minus.min().item(),
            "violation_plus_sum": violation_plus.sum().item(),
            "violation_minus_sum": violation_minus.sum().item(),
        }

    def get_lambda_multipliers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._lambda_plus.clone(), self._lambda_minus.clone()

    def compute_router_z_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        log_z = torch.logsumexp(gate_logits, dim=-1)
        loss = (log_z ** 2).mean()
        return loss

    def get_state(self) -> LagrangianState:
        return LagrangianState(
            lambda_plus=self._lambda_plus.clone(),
            lambda_minus=self._lambda_minus.clone(),
            momentum_plus=self._momentum_plus.clone(),
            momentum_minus=self._momentum_minus.clone(),
            constraint_violation_plus=self._last_violation_plus.clone(),
            constraint_violation_minus=self._last_violation_minus.clone(),
            aug_lagrangian_value=0.0,
            penalty_term=0.0,
        )

    def state_dict(self, *args, **kwargs) -> Dict:
        state = super().state_dict(*args, **kwargs)
        extra = {
            "_step_count": self._step_count,
            "_last_violation_plus": self._last_violation_plus.clone().cpu(),
            "_last_violation_minus": self._last_violation_minus.clone().cpu(),
        }
        state.update(extra)
        return state

    def load_state_dict(self, state_dict: Dict, *args, **kwargs):
        self._step_count = state_dict.pop("_step_count", 0)
        self._last_violation_plus = state_dict.pop("_last_violation_plus", torch.zeros(self.num_experts)).to(self.device)
        self._last_violation_minus = state_dict.pop("_last_violation_minus", torch.zeros(self.num_experts)).to(self.device)
        return super().load_state_dict(state_dict, *args, **kwargs)
