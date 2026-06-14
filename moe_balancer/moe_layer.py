from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MoEConfig, BalancerConfig, DistributedConfig
from .utils import get_logger, load_balance_metrics, all_reduce_mean
from .expert import ExpertGroup
from .utilization_tracker import MovingAverageUtilizationTracker, UtilizationStats
from .similarity import ExpertSimilarityClusterer, ClusteringResult
from .lagrangian import LagrangianBalancer
from .gate import SmartMoEGate, GateOutput

logger = get_logger(__name__)


@dataclass
class MoELayerOutput:
    hidden_states: torch.Tensor
    gate_output: GateOutput
    utilization_stats: Optional[UtilizationStats]
    clustering_result: Optional[ClusteringResult]
    total_aux_loss: torch.Tensor
    metrics: Dict[str, float]


class DistributedMoELayer(nn.Module):
    def __init__(
        self,
        moe_config: MoEConfig,
        balancer_config: BalancerConfig,
        dist_config: Optional[DistributedConfig] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.balancer_config = balancer_config
        self.dist_config = dist_config or DistributedConfig()
        self.device = device or torch.device("cpu")

        self.num_experts = moe_config.num_experts
        self.top_k = moe_config.top_k
        self.model_dim = moe_config.model_dim

        self.gate = SmartMoEGate(moe_config, balancer_config, self.device)
        self.experts = ExpertGroup(moe_config, self.dist_config)

        self.utilization_tracker = MovingAverageUtilizationTracker(
            moe_config, balancer_config, self.dist_config, self.device
        )
        self.clusterer = ExpertSimilarityClusterer(moe_config, balancer_config, self.device)
        self.lagrangian_balancer = LagrangianBalancer(
            moe_config.num_experts, balancer_config, self.device
        )

        self.input_ln = nn.LayerNorm(self.model_dim, eps=1e-5)

        self._global_step = 0
        self._last_clustering: Optional[ClusteringResult] = None

    def to(self, device: torch.device, *args, **kwargs):
        self.device = device
        super().to(device, *args, **kwargs)
        self.utilization_tracker.to(device)
        self.clusterer.to(device)
        self.lagrangian_balancer.to(device)
        return self

    def _maybe_update_clustering(self):
        if self.clusterer.needs_update(self._global_step):
            try:
                expert_reps = self.experts.get_expert_representations()
                self.clusterer.compute_similarity_from_parameters(expert_reps)
                clustering = self.clusterer.cluster_experts_kmeans()
                self._last_clustering = clustering
                self.clusterer.mark_updated(self._global_step)
                logger.info(
                    f"Step {self._global_step}: Updated clustering, "
                    f"num_clusters={clustering.num_clusters}, "
                    f"cluster_size_std={clustering.cluster_sizes.float().std().item():.2f}"
                )
            except Exception as e:
                logger.warning(f"Clustering update failed at step {self._global_step}: {e}")

    def _track_utilization(self, gate_out: GateOutput):
        batch_size, seq_len, top_k = gate_out.expert_weights.shape
        flat_weights = gate_out.expert_weights.reshape(-1, top_k)
        flat_indices = gate_out.expert_indices.reshape(-1, top_k)
        self.utilization_tracker.update(flat_weights, flat_indices)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> MoELayerOutput:
        self._global_step += 1

        residual = hidden_states
        hidden_states = self.input_ln(hidden_states)

        gate_out: GateOutput = self.gate(
            hidden_states,
            tracker=self.utilization_tracker,
            clusterer=self.clusterer,
            lagrangian=self.lagrangian_balancer,
            training=training,
        )

        expert_indices = gate_out.expert_indices
        expert_weights = gate_out.expert_weights

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            expert_weights = expert_weights * mask
            total_w = expert_weights.sum(dim=-1, keepdim=True)
            expert_weights = expert_weights / total_w.clamp_min(1e-9)

        expert_output, _ = self.experts(
            hidden_states,
            expert_indices,
            expert_weights,
        )

        output = residual + expert_output

        total_aux_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        if training:
            total_aux_loss = total_aux_loss + gate_out.aux_loss

            if self._last_clustering is not None:
                stats = self.utilization_tracker.get_stats()
                cluster_penalty = self.clusterer.compute_cluster_balance_penalty(
                    stats.moving_avg_utilization
                )
                total_aux_loss = total_aux_loss + self.balancer_config.aux_loss_weight * cluster_penalty.detach()

        metrics: Dict[str, float] = {}
        util_stats = None

        if gate_out.routing_stats:
            for k, v in gate_out.routing_stats.items():
                metrics[f"routing/{k}"] = v if isinstance(v, float) else (
                    v.item() if hasattr(v, "item") else float(v)
                )

        if training and self._global_step % 10 == 0:
            try:
                util_stats = self.utilization_tracker.get_stats()
                metrics["util/cv_sq"] = util_stats.utilization_cv ** 2
                metrics["util/balance_ratio"] = util_stats.balance_ratio
                metrics["util/low_expert_count"] = len(util_stats.low_utilization_experts)
                metrics["util/high_expert_count"] = len(util_stats.high_utilization_experts)
                metrics["util/variance"] = util_stats.utilization_variance
                metrics["util/mean"] = util_stats.moving_avg_utilization.mean().item()

                batch_size, seq_len, top_k = expert_weights.shape
                flat_w = expert_weights.reshape(-1, top_k).detach()
                lb_metrics = load_balance_metrics(flat_w, self.num_experts)
                for k, v in lb_metrics.items():
                    metrics[f"load_balance/{k}"] = v
            except Exception as e:
                logger.warning(f"Metrics computation failed: {e}")

        if training and gate_out.loss_info:
            for k, v in gate_out.loss_info.items():
                metrics[f"loss/{k}"] = v if isinstance(v, float) else (
                    v.item() if hasattr(v, "item") else float(v)
                )

        return MoELayerOutput(
            hidden_states=output,
            gate_output=gate_out,
            utilization_stats=util_stats,
            clustering_result=self._last_clustering,
            total_aux_loss=total_aux_loss if isinstance(total_aux_loss, torch.Tensor)
                else torch.tensor(total_aux_loss, device=self.device),
            metrics=metrics,
        )

    def get_balance_report(self) -> Dict[str, Any]:
        stats = self.utilization_tracker.get_stats()
        util = stats.moving_avg_utilization.cpu()
        long_tail_count = (util < util.mean() * 0.3).sum().item()
        report = {
            "step": self._global_step,
            "total_tokens": stats.total_tokens_processed,
            "cv_sq": stats.utilization_cv ** 2,
            "util_cv": stats.utilization_cv,
            "utilization_cv": stats.utilization_cv,
            "balance_ratio": stats.balance_ratio,
            "util_max": util.max().item(),
            "util_min": util.min().item(),
            "util_mean": util.mean().item(),
            "util_std": util.std().item(),
            "util_variance": stats.utilization_variance,
            "underutilized_experts": stats.low_utilization_experts.tolist(),
            "overutilized_experts": stats.high_utilization_experts.tolist(),
            "num_under": len(stats.low_utilization_experts),
            "num_over": len(stats.high_utilization_experts),
            "long_tail_count": long_tail_count,
            "lambda_stats": {
                "plus_mean": self.lagrangian_balancer._lambda_plus.mean().item(),
                "plus_max": self.lagrangian_balancer._lambda_plus.max().item(),
                "minus_mean": self.lagrangian_balancer._lambda_minus.mean().item(),
                "minus_max": self.lagrangian_balancer._lambda_minus.max().item(),
            },
        }
        if self._last_clustering is not None:
            report["num_clusters"] = self._last_clustering.num_clusters
            report["cluster_size_std"] = self._last_clustering.cluster_sizes.float().std().item()
        return report

    def reset_tracking(self):
        self.utilization_tracker.reset()
        self._global_step = 0

    def step_balancers(
        self,
        gate_output: GateOutput,
    ) -> Dict[str, float]:
        step_util = gate_output.step_utilization
        expert_indices = gate_output.expert_indices
        expert_weights = gate_output.expert_weights

        batch_size, seq_len, top_k = expert_weights.shape
        flat_weights = expert_weights.reshape(-1, top_k).detach()
        flat_indices = expert_indices.reshape(-1, top_k).detach()

        self.utilization_tracker.update(flat_weights, flat_indices)

        lagrangian_info = self.lagrangian_balancer.step_multipliers(step_util)

        clustering_info = {}
        if self.clusterer.needs_update(self._global_step):
            try:
                expert_reps = self.experts.get_expert_representations()
                self.clusterer.compute_similarity_from_parameters(expert_reps)
                clustering = self.clusterer.cluster_experts_kmeans()
                self._last_clustering = clustering
                self.clusterer.mark_updated(self._global_step)
                clustering_info = {
                    "clusterer/updated": True,
                    "clusterer/num_clusters": clustering.num_clusters,
                    "clusterer/size_std": clustering.cluster_sizes.float().std().item(),
                }
            except Exception as e:
                logger.warning(f"Clustering update failed at step {self._global_step}: {e}")
                clustering_info = {"clusterer/updated": False, "clusterer/error": str(e)}

        update_info = {**lagrangian_info, **clustering_info}
        return update_info
