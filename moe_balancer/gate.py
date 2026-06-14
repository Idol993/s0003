from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MoEConfig, BalancerConfig
from .utils import safe_softmax, gumbel_softmax_sample, get_logger, entropy
from .utilization_tracker import MovingAverageUtilizationTracker
from .similarity import ExpertSimilarityClusterer
from .lagrangian import LagrangianBalancer

logger = get_logger(__name__)


@dataclass
class GateOutput:
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    raw_logits: torch.Tensor
    adjusted_logits: torch.Tensor
    force_assign_mask: torch.Tensor
    cluster_redirect_mask: torch.Tensor
    aux_loss: torch.Tensor
    step_utilization: torch.Tensor
    loss_info: Dict[str, float]
    routing_stats: Dict[str, Any]


class SmartMoEGate(nn.Module):
    def __init__(
        self,
        moe_config: MoEConfig,
        balancer_config: BalancerConfig,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.balancer_config = balancer_config
        self.device = device or torch.device("cpu")

        self.num_experts = moe_config.num_experts
        self.top_k = moe_config.top_k
        self.model_dim = moe_config.model_dim

        self.gate_proj = nn.Sequential(
            nn.Linear(self.model_dim, moe_config.gate_hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(moe_config.gate_hidden_dim, self.num_experts, bias=True),
        )

        self.noise_linear = nn.Linear(self.model_dim, self.num_experts, bias=False)
        nn.init.zeros_(self.noise_linear.weight)

        self.register_buffer("_util_bias", torch.zeros(self.num_experts, device=self.device))
        self._init_gate_weights()

    def to(self, device: torch.device, *args, **kwargs):
        self.device = device
        return super().to(device, *args, **kwargs)

    def _init_gate_weights(self):
        for m in self.gate_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _add_gate_noise(
        self,
        logits: torch.Tensor,
        x: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        if not training:
            return logits

        noise_std = self.moe_config.gate_noise_std
        if noise_std <= 0:
            return logits

        adaptive_noise = self.noise_linear(x)
        noise = torch.randn_like(logits) * (noise_std + adaptive_noise.abs())
        return logits + noise

    def _apply_utilization_bias(
        self,
        logits: torch.Tensor,
        tracker: MovingAverageUtilizationTracker,
    ) -> torch.Tensor:
        if tracker.step_count < tracker.warmup_steps:
            warmup_ratio = max(tracker.step_count, 1) / max(tracker.warmup_steps, 1)
        else:
            warmup_ratio = 1.0

        priority_scores = tracker.get_utilization_priority_scores()
        lambda_plus, lambda_minus = None, None

        max_bias = 0.5
        bias = priority_scores * max_bias * warmup_ratio

        bias = bias.unsqueeze(0).expand(logits.shape[0], -1)
        adjusted = logits + bias

        self._util_bias.copy_((priority_scores * max_bias * warmup_ratio).detach())

        return adjusted

    def _get_force_assign_tokens(
        self,
        logits: torch.Tensor,
        tracker: MovingAverageUtilizationTracker,
        clusterer: Optional[ExpertSimilarityClusterer],
        training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = logits.shape[0]
        top_k = self.top_k

        force_assign_mask = torch.zeros(
            batch_size, top_k, dtype=torch.bool, device=self.device
        )
        force_assign_experts = torch.full(
            (batch_size, top_k), -1, dtype=torch.long, device=self.device
        )

        if not training:
            return force_assign_mask, force_assign_experts

        if tracker is None:
            return force_assign_mask, force_assign_experts

        under_experts = tracker.get_underutilized_experts()
        if len(under_experts) == 0:
            return force_assign_mask, force_assign_experts

        max_forced = int(batch_size * self.balancer_config.force_assign_ratio)
        max_forced_per_expert = max(1, int(max_forced / max(len(under_experts), 1)))

        softmax_probs = F.softmax(logits, dim=-1)

        for under_idx_tensor in under_experts:
            under_idx = under_idx_tensor.item()

            probs_for_expert = softmax_probs[:, under_idx]

            base_rank = torch.argsort(probs_for_expert, descending=True)

            if clusterer is not None and clusterer._cluster_assignments is not None:
                cluster_idx = clusterer.get_cluster_for_expert(under_idx)
                if cluster_idx is not None:
                    cluster_members = torch.tensor(
                        clusterer.get_cluster_members(cluster_idx),
                        device=self.device,
                    )
                    if len(cluster_members) > 1:
                        cluster_probs = softmax_probs.index_select(1, cluster_members).sum(dim=-1)
                        cluster_rank = torch.argsort(cluster_probs, descending=True)
                        combined_rank = 0.7 * base_rank.float() + 0.3 * cluster_rank.float()
                        candidates = torch.argsort(combined_rank)
                    else:
                        candidates = base_rank
                else:
                    candidates = base_rank
            else:
                candidates = base_rank

            assigned_count = 0
            for cand_idx in candidates:
                if assigned_count >= max_forced_per_expert:
                    break
                cand_idx_int = cand_idx.item()

                slot = -1
                for k in range(top_k):
                    if not force_assign_mask[cand_idx_int, k]:
                        slot = k
                        break

                if slot >= 0:
                    force_assign_mask[cand_idx_int, slot] = True
                    force_assign_experts[cand_idx_int, slot] = under_idx
                    assigned_count += 1

        return force_assign_mask, force_assign_experts

    def _apply_cluster_redirect(
        self,
        top_indices: torch.Tensor,
        tracker: MovingAverageUtilizationTracker,
        clusterer: Optional[ExpertSimilarityClusterer],
        training: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, top_k = top_indices.shape
        redirect_mask = torch.zeros_like(top_indices, dtype=torch.bool)

        if not training or clusterer is None:
            return top_indices.clone(), redirect_mask

        over_experts = tracker.get_overutilized_experts()
        if len(over_experts) == 0:
            return top_indices.clone(), redirect_mask

        new_indices = top_indices.clone()

        for batch_i in range(batch_size):
            for k in range(top_k):
                expert = top_indices[batch_i, k].item()
                if expert in over_experts.tolist():
                    cluster_idx = clusterer.get_cluster_for_expert(expert)
                    if cluster_idx is not None:
                        members = clusterer.get_cluster_members(cluster_idx)
                        if len(members) > 1:
                            bc_util = tracker.get_bias_corrected_utilization()
                            member_utils = bc_util[
                                torch.tensor(members, device=self.device)
                            ]
                            best_member_idx = members[torch.argmin(member_utils).item()]
                            new_indices[batch_i, k] = best_member_idx
                            redirect_mask[batch_i, k] = True

        return new_indices, redirect_mask

    def forward(
        self,
        x: torch.Tensor,
        tracker: Optional[MovingAverageUtilizationTracker] = None,
        clusterer: Optional[ExpertSimilarityClusterer] = None,
        lagrangian: Optional[LagrangianBalancer] = None,
        training: bool = True,
    ) -> GateOutput:
        batch_size, seq_len, model_dim = x.shape
        num_tokens = batch_size * seq_len
        x_flat = x.view(num_tokens, model_dim)

        raw_logits = self.gate_proj(x_flat)

        noised_logits = self._add_gate_noise(raw_logits, x_flat, training)

        adjusted_logits = noised_logits
        if tracker is not None and tracker.step_count > 10:
            adjusted_logits = self._apply_utilization_bias(adjusted_logits, tracker)

        if lagrangian is not None and lagrangian._step_count > 0:
            lambda_plus, lambda_minus = lagrangian.get_lambda_multipliers()
            lambda_bias = (lambda_minus - lambda_plus) * 0.01
            adjusted_logits = adjusted_logits + lambda_bias.unsqueeze(0)

        gumbel_weights = gumbel_softmax_sample(adjusted_logits, temperature=1.0, hard=False, dim=-1)

        top_values, top_indices = torch.topk(gumbel_weights, self.top_k, dim=-1)

        force_mask, force_experts = self._get_force_assign_tokens(
            adjusted_logits, tracker, clusterer, training
        )
        for k in range(self.top_k):
            k_mask = force_mask[:, k]
            if k_mask.any():
                expert_ids = force_experts[k_mask, k]
                top_indices[k_mask, k] = expert_ids
                top_values[k_mask, k] = 1.0 / self.top_k

        top_indices, redirect_mask = self._apply_cluster_redirect(
            top_indices, tracker, clusterer, training
        )

        weights_src = gumbel_softmax_sample(adjusted_logits, temperature=0.5, hard=False, dim=-1)
        batch_arange = torch.arange(num_tokens, device=self.device).unsqueeze(-1)
        expert_weights = weights_src[batch_arange, top_indices]

        total_weights = expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights / total_weights.clamp_min(1e-9)

        loss_info: Dict[str, float] = {}
        routing_stats: Dict[str, Any] = {}
        aux_loss = torch.tensor(0.0, device=self.device)

        if lagrangian is not None and training:
            penalty, lb_info = lagrangian.compute_load_balance_penalty(
                adjusted_logits,
                expert_weights=None,
                expert_utilization=None,
            )
            z_loss = lagrangian.compute_router_z_loss(raw_logits)
            total_aux = penalty + self.balancer_config.router_z_loss_weight * z_loss
            bias_correction = lagrangian.compute_implicit_gradient_bias_correction(
                weights_src, force_mask
            )
            total_aux = total_aux + self.balancer_config.aux_loss_weight * bias_correction

            aux_loss = total_aux

            loss_info["load_balance_penalty"] = lb_info["total_penalty"]
            loss_info["router_z_loss"] = z_loss.item()
            loss_info["bias_correction"] = bias_correction.item() if isinstance(bias_correction, torch.Tensor) else bias_correction
            loss_info["total_aux_loss"] = total_aux.item() if isinstance(total_aux, torch.Tensor) else total_aux
            loss_info.update(lb_info)

        expert_indices = top_indices.view(batch_size, seq_len, self.top_k)
        expert_weights_out = expert_weights.view(batch_size, seq_len, self.top_k)
        raw_logits_out = raw_logits.view(batch_size, seq_len, self.num_experts)
        adj_logits_out = adjusted_logits.view(batch_size, seq_len, self.num_experts)
        force_mask_out = force_mask.view(batch_size, seq_len, self.top_k)
        redirect_mask_out = redirect_mask.view(batch_size, seq_len, self.top_k)

        with torch.no_grad():
            full_weights = F.softmax(adjusted_logits, dim=-1)
            step_util = full_weights.mean(dim=0).detach()
            top1_idx = full_weights.argmax(dim=-1)
            unique, counts = torch.unique(top1_idx, return_counts=True)
            routing_stats["num_unique_routed"] = len(unique)
            if len(counts) > 0:
                routing_stats["top1_max_ratio"] = (counts.max() / counts.sum()).item()
                routing_stats["entropy"] = entropy(full_weights).mean().item()
            routing_stats["force_assign_ratio"] = force_mask.float().mean().item()
            routing_stats["cluster_redirect_ratio"] = redirect_mask.float().mean().item()

        return GateOutput(
            expert_indices=expert_indices,
            expert_weights=expert_weights_out,
            raw_logits=raw_logits_out,
            adjusted_logits=adj_logits_out,
            force_assign_mask=force_mask_out,
            cluster_redirect_mask=redirect_mask_out,
            aux_loss=aux_loss,
            step_utilization=step_util,
            loss_info=loss_info,
            routing_stats=routing_stats,
        )
