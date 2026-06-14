import logging
import math
import os
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist


def get_logger(name: str, log_level: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    level = (log_level or os.environ.get("MOE_LOG_LEVEL", "INFO")).upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger


def safe_softmax(logits: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    logits = logits - logits.max(dim=dim, keepdim=True)[0]
    exp_logits = torch.exp(logits.clamp_min(-30.0))
    sum_exp = exp_logits.sum(dim=dim, keepdim=True) + eps
    return exp_logits / sum_exp


def gumbel_softmax_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = False,
    dim: int = -1,
    eps: float = 1e-9,
) -> torch.Tensor:
    uniform = torch.rand_like(logits).clamp_min(eps).clamp_max(1.0 - eps)
    gumbel = -torch.log(-torch.log(uniform))
    noisy_logits = (logits + gumbel) / max(temperature, eps)
    y_soft = safe_softmax(noisy_logits, dim=dim, eps=eps)
    if hard:
        index = y_soft.max(dim=dim, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
        ret = y_hard.detach() - y_soft.detach() + y_soft
        return ret
    return y_soft


def compute_cosine_similarity(
    x: torch.Tensor, y: Optional[torch.Tensor] = None, eps: float = 1e-8
) -> torch.Tensor:
    if y is None:
        y = x
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    return torch.matmul(x_norm, y_norm.transpose(-2, -1))


def all_reduce_sum(tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor.clone()
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor


def all_reduce_mean(tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return tensor.clone()
    world_size = dist.get_world_size(group) if group else dist.get_world_size()
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    return tensor / world_size


def entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    p = p.clamp_min(eps)
    return -(p * torch.log(p)).sum(dim=dim)


def variance(x: torch.Tensor, dim: int = -1, unbiased: bool = True) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    diff = x - mean
    n = x.shape[dim] - int(unbiased)
    return (diff * diff).sum(dim=dim) / max(n, 1)


def load_balance_metrics(
    expert_weights: torch.Tensor,
    num_experts: int,
    eps: float = 1e-9,
) -> dict:
    batch_size = expert_weights.shape[0]
    expert_loads = expert_weights.sum(dim=0)
    total_load = expert_loads.sum() + eps
    frac = expert_loads / total_load
    ideal_frac = 1.0 / num_experts
    cv_sq = num_experts * ((frac - ideal_frac) ** 2).sum()
    balance_ratio = frac.max() / (frac.min() + eps)
    active_experts = (expert_loads > 0).float().mean()
    top1_expert_share = expert_loads.max() / total_load
    return {
        "cv_sq": cv_sq.item(),
        "balance_ratio": balance_ratio.item(),
        "active_experts": active_experts.item(),
        "top1_expert_share": top1_expert_share.item(),
        "mean_load": (total_load / batch_size / num_experts).item(),
        "entropy": entropy(frac.unsqueeze(0)).mean().item(),
    }
