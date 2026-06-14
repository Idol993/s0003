from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .config import MoEConfig, DistributedConfig


class ExpertLayer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(model_dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, model_dim, bias=True)
        self.w3 = nn.Linear(model_dim, hidden_dim, bias=True)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.activation = self._get_activation(activation)

        self._init_weights()

    def _get_activation(self, name: str):
        act_map = {
            "silu": F.silu,
            "gelu": F.gelu,
            "relu": F.relu,
            "leaky_relu": F.leaky_relu,
        }
        if name not in act_map:
            raise ValueError(f"Unknown activation: {name}")
        return act_map[name]

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w1.weight, gain=0.5)
        nn.init.xavier_uniform_(self.w3.weight, gain=0.5)
        nn.init.xavier_uniform_(self.w2.weight, gain=0.5)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)
        nn.init.zeros_(self.w3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.activation(self.w1(x))
        up = self.w3(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        out = self.w2(hidden)
        return out


class ExpertGroup(nn.Module):
    def __init__(
        self,
        moe_config: MoEConfig,
        dist_config: Optional[DistributedConfig] = None,
    ):
        super().__init__()
        self.moe_config = moe_config
        self.dist_config = dist_config or DistributedConfig()

        total_experts = moe_config.num_experts
        if moe_config.use_expert_parallel and self.dist_config.expert_parallel_size > 1:
            assert total_experts % self.dist_config.expert_parallel_size == 0, (
                f"Total experts {total_experts} must be divisible by expert parallel size "
                f"{self.dist_config.expert_parallel_size}"
            )
            self.num_local_experts = total_experts // self.dist_config.expert_parallel_size
            self.local_start_idx = self.dist_config.expert_parallel_rank * self.num_local_experts
            self.local_end_idx = self.local_start_idx + self.num_local_experts
        else:
            self.num_local_experts = total_experts
            self.local_start_idx = 0
            self.local_end_idx = total_experts

        self.experts_per_group = min(moe_config.experts_per_group, self.num_local_experts)
        self.num_groups = (self.num_local_experts + self.experts_per_group - 1) // self.experts_per_group

        self.groups = nn.ModuleList()
        for g in range(self.num_groups):
            start = g * self.experts_per_group
            end = min(start + self.experts_per_group, self.num_local_experts)
            group_experts = nn.ModuleList([
                ExpertLayer(
                    moe_config.model_dim,
                    moe_config.expert_hidden_dim,
                    moe_config.expert_activation,
                    moe_config.expert_dropout,
                )
                for _ in range(end - start)
            ])
            self.groups.append(group_experts)

        self._total_params = sum(p.numel() for p in self.parameters())

    @property
    def total_parameters(self) -> int:
        return self._total_params

    def get_expert_representations(self) -> torch.Tensor:
        reps = []
        for group in self.groups:
            for expert in group:
                rep = torch.cat([
                    expert.w1.weight.data.view(-1),
                    expert.w3.weight.data.view(-1),
                    expert.w2.weight.data.view(-1),
                ], dim=0)
                reps.append(rep)
        return torch.stack(reps, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, model_dim = x.shape
        num_tokens = batch_size * seq_len
        x_flat = x.view(num_tokens, model_dim)
        top_k = expert_indices.shape[-1]

        output_flat = torch.zeros_like(x_flat)
        indices_flat = expert_indices.view(num_tokens, top_k)
        weights_flat = expert_weights.view(num_tokens, top_k) if expert_weights is not None else None

        for local_group_idx, group in enumerate(self.groups):
            group_start = local_group_idx * self.experts_per_group
            for in_group_idx, expert in enumerate(group):
                expert_local_idx = group_start + in_group_idx
                expert_global_idx = self.local_start_idx + expert_local_idx

                matches = (indices_flat == expert_global_idx)
                if not matches.any():
                    continue

                for k in range(top_k):
                    mask_k = matches[:, k]
                    if not mask_k.any():
                        continue

                    token_ids = torch.nonzero(mask_k, as_tuple=False).squeeze(-1)
                    expert_inputs = x_flat.index_select(0, token_ids)
                    expert_outputs = expert(expert_inputs)

                    if weights_flat is not None:
                        w = weights_flat.index_select(0, token_ids)[:, k].unsqueeze(-1)
                        expert_outputs = expert_outputs * w

                    output_flat.index_add_(0, token_ids, expert_outputs)

        if self.moe_config.use_expert_parallel and self.dist_config.expert_parallel_size > 1:
            if dist.is_initialized():
                dist.all_reduce(output_flat, op=dist.ReduceOp.SUM)

        output = output_flat.view(batch_size, seq_len, model_dim)
        return output, expert_weights

    def batched_forward_single_expert(
        self,
        x_list: List[torch.Tensor],
        expert_global_idx: int,
    ) -> torch.Tensor:
        if not (self.local_start_idx <= expert_global_idx < self.local_end_idx):
            return torch.zeros_like(torch.cat(x_list, dim=0) if x_list else torch.empty(0))

        local_idx = expert_global_idx - self.local_start_idx
        group_idx = local_idx // self.experts_per_group
        in_group_idx = local_idx % self.experts_per_group
        expert = self.groups[group_idx][in_group_idx]

        if not x_list:
            return torch.empty(0, self.moe_config.model_dim, device=next(self.parameters()).device)

        tokens = torch.cat(x_list, dim=0)
        return expert(tokens)
