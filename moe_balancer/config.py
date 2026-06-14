from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import torch


@dataclass
class MoEConfig:
    num_experts: int = 1024
    top_k: int = 2
    model_dim: int = 1024
    expert_hidden_dim: int = 4096
    expert_activation: str = "silu"
    expert_dropout: float = 0.0
    gate_hidden_dim: int = 512
    gate_noise_std: float = 1.0
    use_expert_parallel: bool = True
    experts_per_group: int = 64


@dataclass
class DistributedConfig:
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    expert_parallel_size: int = 1
    data_parallel_size: int = 1
    backend: str = "nccl"
    dist_url: str = "env://"
    use_distributed: bool = False

    @property
    def expert_parallel_rank(self) -> int:
        if self.expert_parallel_size <= 1:
            return 0
        return self.rank % self.expert_parallel_size

    @property
    def data_parallel_rank(self) -> int:
        if self.data_parallel_size <= 1:
            return 0
        return self.rank // self.expert_parallel_size


@dataclass
class BalancerConfig:
    utilization_momentum: float = 0.9
    utilization_warmup_steps: int = 100
    utilization_threshold_low: float = 0.1
    utilization_threshold_high: float = 0.5
    force_assign_ratio: float = 0.15
    max_force_assign_ratio: float = 0.3
    aux_loss_weight: float = 0.01
    lagrangian_init_lambda: float = 1.0
    lagrangian_lr: float = 0.01
    lagrangian_momentum: float = 0.9
    lagrangian_weight_decay: float = 0.0
    lagrangian_max_lambda: float = 10.0
    lagrangian_min_lambda: float = 0.01
    load_balance_loss_weight: float = 1.0
    router_z_loss_weight: float = 0.001
    cluster_update_interval: int = 50
    num_clusters: int = 64
    similarity_top_k: int = 8
    grad_clip_for_gate: float = 1.0
    bias_correction_factor: float = 0.9
