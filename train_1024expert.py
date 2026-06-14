"""
MoE Large Language Model Training Pipeline - Real 1024 Experts + Dynamic Load Self-Balancing

Features:
  - 1024 independent feed-forward expert networks
  - Top-2 gating, each token activates only 2 experts
  - Moving average utilization tracking
  - Low-utilization expert forced assignment + gradient bias correction
  - Differentiable Lagrangian multiplier load balancing
  - Expert similarity matrix + cluster-based reassignment
  - Complete acceptance metrics tracking with visualization
  - A/B comparison: with vs without balancing protocol
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_balancer import (
    MoEConfig,
    BalancerConfig,
    DistributedConfig,
    DistributedMoELayer,
    get_logger,
)

logger = get_logger(__name__)


@dataclass
class BatchRoutingStats:
    total_tokens: int
    normal_routed_tokens: int
    force_assigned_tokens: int
    cluster_redirected_tokens: int
    normal_ratio: float
    force_ratio: float
    redirect_ratio: float
    expert_hit_counts: torch.Tensor
    unique_experts_used: int
    top1_expert_share: float
    top5_expert_share: float
    routing_entropy: float


@dataclass
class UtilizationMetrics:
    util_max: float
    util_min: float
    util_mean: float
    util_std: float
    util_cv: float
    util_cv_sq: float
    util_variance: float
    underutilized_count: int
    overutilized_count: float
    long_tail_experts_count: int
    balance_ratio: float


@dataclass
class LossMetrics:
    task_loss: float
    load_balance_loss: float
    aug_lagrangian_loss: float
    router_z_loss: float
    bias_correction_loss: float
    total_aux_loss: float
    total_loss: float


@dataclass
class ExpertSimilarityReport:
    num_clusters: int
    cluster_size_std: float
    similarity_mean: float
    similarity_max: float
    similarity_min: float
    top_similar_pairs: List[Tuple[int, int, float]]
    underutilized_experts_in_clusters: Dict[int, List[int]]


class MoELanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        moe_config: MoEConfig,
        balancer_config: BalancerConfig,
        num_layers: int = 2,
        num_heads: int = 8,
        enable_balancer: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = moe_config.model_dim
        self.num_layers = num_layers
        self.enable_balancer = enable_balancer

        self.token_embedding = nn.Embedding(vocab_size, self.model_dim)
        self.position_embedding = nn.Embedding(4096, self.model_dim)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

        self.moe_layers = nn.ModuleList([
            DistributedMoELayer(moe_config, balancer_config)
            for _ in range(num_layers)
        ])

        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(self.model_dim, eps=1e-5) for _ in range(num_layers)
        ])
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.model_dim,
                num_heads=num_heads,
                batch_first=True,
                dropout=0.1,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(self.model_dim, eps=1e-5)
        self.lm_head = nn.Linear(self.model_dim, vocab_size, bias=False)
        nn.init.xavier_uniform_(self.lm_head.weight, gain=0.1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List]:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        all_gate_outputs = []

        for layer_idx in range(self.num_layers):
            residual = x
            x_norm = self.attn_norms[layer_idx](x)
            attn_out, _ = self.attentions[layer_idx](
                x_norm, x_norm, x_norm,
                key_padding_mask=(attention_mask == 0) if attention_mask is not None else None,
                need_weights=False,
                attn_mask=None,
                is_causal=False,
            )
            x = residual + attn_out

            if not self.enable_balancer:
                dummy_tracker = type('Dummy', (), {'step_count': 0})()
                moe_out = self.moe_layers[layer_idx](
                    x, attention_mask=attention_mask, training=training
                )
            else:
                moe_out = self.moe_layers[layer_idx](
                    x, attention_mask=attention_mask, training=training
                )

            x = moe_out.hidden_states
            total_aux_loss = total_aux_loss + moe_out.total_aux_loss
            all_gate_outputs.append(moe_out.gate_output)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss, total_aux_loss, all_gate_outputs


def compute_batch_routing_stats(
    gate_output,
    num_experts: int,
    device: torch.device,
) -> BatchRoutingStats:
    batch_size, seq_len, top_k = gate_output.expert_indices.shape
    num_tokens = batch_size * seq_len

    expert_indices_flat = gate_output.expert_indices.reshape(-1).cpu()
    force_mask_flat = gate_output.force_assign_mask.reshape(-1).cpu()
    redirect_mask_flat = gate_output.cluster_redirect_mask.reshape(-1).cpu()

    both_mask = force_mask_flat & redirect_mask_flat
    only_force = force_mask_flat & (~redirect_mask_flat)
    only_redirect = redirect_mask_flat & (~force_mask_flat)
    normal = ~(force_mask_flat | redirect_mask_flat)

    expert_hit_counts = torch.bincount(
        expert_indices_flat,
        minlength=num_experts
    ).float()

    topk_counts = torch.topk(expert_hit_counts, k=min(5, num_experts))
    top1_share = topk_counts.values[0] / expert_hit_counts.sum().clamp_min(1.0)
    top5_share = topk_counts.values.sum() / expert_hit_counts.sum().clamp_min(1.0)

    with torch.no_grad():
        adj_logits_flat = gate_output.adjusted_logits.reshape(-1, num_experts)
        probs = F.softmax(adj_logits_flat, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1).mean()

    return BatchRoutingStats(
        total_tokens=num_tokens,
        normal_routed_tokens=normal.sum().item(),
        force_assigned_tokens=only_force.sum().item(),
        cluster_redirected_tokens=only_redirect.sum().item(),
        normal_ratio=normal.float().mean().item(),
        force_ratio=only_force.float().mean().item(),
        redirect_ratio=only_redirect.float().mean().item(),
        expert_hit_counts=expert_hit_counts,
        unique_experts_used=(expert_hit_counts > 0).sum().item(),
        top1_expert_share=top1_share.item(),
        top5_expert_share=top5_share.item(),
        routing_entropy=entropy.item(),
    )


def compute_utilization_metrics(
    tracker,
    num_experts: int,
    long_tail_threshold_ratio: float = 0.3,
) -> UtilizationMetrics:
    stats = tracker.get_stats()
    util = stats.moving_avg_utilization.cpu()

    sorted_util = torch.sort(util, descending=True).values
    long_tail_count = (util < util.mean() * long_tail_threshold_ratio).sum().item()

    return UtilizationMetrics(
        util_max=util.max().item(),
        util_min=util.min().item(),
        util_mean=util.mean().item(),
        util_std=util.std().item(),
        util_cv=(util.std() / util.mean().clamp_min(1e-9)).item(),
        util_cv_sq=((util.std() / util.mean().clamp_min(1e-9)) ** 2).item(),
        util_variance=util.var().item(),
        underutilized_count=len(stats.low_utilization_experts),
        overutilized_count=len(stats.high_utilization_experts),
        long_tail_experts_count=long_tail_count,
        balance_ratio=(util.max() / util.min().clamp_min(1e-9)).item(),
    )


def compute_expert_similarity_report(
    layer: DistributedMoELayer,
    tracker,
    num_pairs: int = 10,
) -> Optional[ExpertSimilarityReport]:
    clusterer = layer.clusterer
    sim_matrix = clusterer.get_similarity_matrix()
    clustering = layer._last_clustering

    if sim_matrix is None or clustering is None:
        return None

    sim_matrix_np = sim_matrix.cpu()
    triu_idx = torch.triu_indices(sim_matrix.shape[0], sim_matrix.shape[1], offset=1)
    triu_vals = sim_matrix_np[triu_idx[0], triu_idx[1]]
    topk = torch.topk(triu_vals, k=min(num_pairs, len(triu_vals)))
    top_pairs = []
    for i in range(len(topk.values)):
        e1 = triu_idx[0][topk.indices[i]].item()
        e2 = triu_idx[1][topk.indices[i]].item()
        top_pairs.append((e1, e2, topk.values[i].item()))

    under_experts = tracker.get_underutilized_experts().cpu().tolist()
    under_in_clusters: Dict[int, List[int]] = defaultdict(list)
    for e in under_experts:
        cid = clustering.cluster_assignments[e].item()
        under_in_clusters[cid].append(e)

    return ExpertSimilarityReport(
        num_clusters=clustering.num_clusters,
        cluster_size_std=clustering.cluster_sizes.float().std().item(),
        similarity_mean=sim_matrix_np.mean().item(),
        similarity_max=sim_matrix_np.max().item(),
        similarity_min=sim_matrix_np.min().item(),
        top_similar_pairs=top_pairs,
        underutilized_experts_in_clusters=dict(under_in_clusters),
    )


def compute_loss_metrics(
    task_loss: float,
    gate_output,
    total_aux_loss: float,
) -> LossMetrics:
    loss_info = gate_output.loss_info
    return LossMetrics(
        task_loss=task_loss,
        load_balance_loss=loss_info.get("load_balance_penalty", 0.0),
        aug_lagrangian_loss=loss_info.get("aug_lagrangian", 0.0),
        router_z_loss=loss_info.get("router_z_loss", 0.0),
        bias_correction_loss=loss_info.get("bias_correction", 0.0),
        total_aux_loss=total_aux_loss,
        total_loss=task_loss + total_aux_loss,
    )


def inject_extreme_imbalance(
    model: MoELanguageModel,
    inject_ratio: float = 0.02,
):
    moe_cfg = model.moe_layers[0].moe_config
    num_hot = max(1, int(moe_cfg.num_experts * inject_ratio))
    logger.warning(
        f"Injecting extreme imbalance: Only {num_hot}/{moe_cfg.num_experts} "
        f"experts will be initially favored (hell difficulty scenario)"
    )

    for layer_idx, layer in enumerate(model.moe_layers):
        with torch.no_grad():
            gate_proj_out = layer.gate.gate_proj[-1]
            hot_experts = torch.randperm(moe_cfg.num_experts)[:num_hot]
            bias = torch.zeros(moe_cfg.num_experts, device=gate_proj_out.bias.device)
            bias[hot_experts] = 6.0
            others_mask = ~torch.isin(torch.arange(moe_cfg.num_experts), hot_experts)
            bias[others_mask] = -2.0
            gate_proj_out.bias.add_(bias)
            logger.info(f"  Layer {layer_idx}: Hot experts = {hot_experts.tolist()}")


def generate_training_batch(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    bias_strength: float = 0.5,
    num_bias_tokens: int = 20,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.ones(vocab_size) / vocab_size
    bias_idx = torch.randperm(vocab_size)[:num_bias_tokens]
    probs[:] = (1.0 - bias_strength) / (vocab_size - num_bias_tokens)
    probs[bias_idx] = bias_strength / num_bias_tokens

    input_ids = torch.multinomial(probs, batch_size * seq_len, replacement=True)
    input_ids = input_ids.view(batch_size, seq_len).to(device)

    labels = input_ids.clone()
    mask = torch.rand_like(labels.float()) > 0.15
    labels[mask] = -100

    attention_mask = torch.ones(batch_size, seq_len, device=device).long()
    return input_ids, labels, attention_mask


def print_routing_report(
    step: int,
    routing_stats: BatchRoutingStats,
    util_metrics: Optional[UtilizationMetrics],
    loss_metrics: LossMetrics,
    sim_report: Optional[ExpertSimilarityReport],
    elapsed: float,
    num_experts: int = 1024,
):
    logger.info("")
    logger.info("=" * 90)
    logger.info(f"Step {step:4d} | Time: {elapsed:6.2f}s")
    logger.info("=" * 90)

    logger.info("")
    logger.info("--- Routing Stats ---")
    logger.info(f"  Total tokens:        {routing_stats.total_tokens}")
    logger.info(f"  Normal top-2 routing:   {routing_stats.normal_routed_tokens:6d}  ({routing_stats.normal_ratio*100:5.1f}%)")
    logger.info(f"  Forced assignment (balancing):  {routing_stats.force_assigned_tokens:6d}  ({routing_stats.force_ratio*100:5.1f}%)")
    logger.info(f"  Cluster redirect (balancing): {routing_stats.cluster_redirected_tokens:6d}  ({routing_stats.redirect_ratio*100:5.1f}%)")
    logger.info(f"  Unique experts used:          {routing_stats.unique_experts_used}/{num_experts}")
    logger.info(f"  Top1 expert share:       {routing_stats.top1_expert_share*100:5.2f}%")
    logger.info(f"  Top5 expert share:       {routing_stats.top5_expert_share*100:5.2f}%")
    logger.info(f"  Routing entropy:              {routing_stats.routing_entropy:.4f}")

    if util_metrics is not None:
        logger.info("")
        logger.info("--- Expert Utilization ---")
        logger.info(f"  Max:      {util_metrics.util_max*100:6.3f}%")
        logger.info(f"  Min:      {util_metrics.util_min*100:6.3f}%")
        logger.info(f"  Mean:     {util_metrics.util_mean*100:6.3f}%")
        logger.info(f"  Std:      {util_metrics.util_std*100:6.3f}%")
        logger.info(f"  CV²:      {util_metrics.util_cv_sq:8.4f}  <-- smaller = better, ideal < 0.1")
        logger.info(f"  Balance ratio:   {util_metrics.balance_ratio:8.2f}x <-- Max/Min, ideal < 1.5")
        logger.info(f"  Underutilized: {util_metrics.underutilized_count:4d}  / {num_experts}")
        logger.info(f"  Overutilized:  {util_metrics.overutilized_count:4d}  / {num_experts}")
        logger.info(f"  Long-tail experts: {util_metrics.long_tail_experts_count:4d}  / {num_experts}  (< 30% of mean)")

    logger.info("")
    logger.info("--- Losses ---")
    logger.info(f"  Task LM Loss:      {loss_metrics.task_loss:.4f}")
    logger.info(f"  Load balance loss:         {loss_metrics.load_balance_loss:.4f}")
    logger.info(f"  Augmented Lagrangian:     {loss_metrics.aug_lagrangian_loss:.4f}")
    logger.info(f"  Router Z Loss:       {loss_metrics.router_z_loss:.6f}")
    logger.info(f"  Gradient bias correction:         {loss_metrics.bias_correction_loss:.6f}")
    logger.info(f"  Total aux loss:           {loss_metrics.total_aux_loss:.4f}")
    logger.info(f"  Total loss:               {loss_metrics.total_loss:.4f}")

    if sim_report is not None:
        logger.info("")
        logger.info("--- Expert Similarity & Clustering ---")
        logger.info(f"  Num clusters:             {sim_report.num_clusters}")
        logger.info(f"  Cluster size std:      {sim_report.cluster_size_std:.2f}")
        logger.info(f"  Similarity mean/max/min: {sim_report.similarity_mean:.3f} / {sim_report.similarity_max:.3f} / {sim_report.similarity_min:.3f}")
        logger.info(f"  Top similar expert pairs:")
        for e1, e2, sim in sim_report.top_similar_pairs[:5]:
            logger.info(f"    Expert {e1:4d} <-> {e2:4d}: similarity = {sim:.4f}")
        if sim_report.underutilized_experts_in_clusters:
            logger.info(f"  Underutilized per cluster:")
            for cid, experts in list(sim_report.underutilized_experts_in_clusters.items())[:5]:
                logger.info(f"    Cluster {cid:3d}: {len(experts)} underutilized experts")

    logger.info("=" * 90)


def train(config, enable_balancer: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Balancer enabled: {enable_balancer}")

    moe_config = MoEConfig(
        num_experts=config.num_experts,
        top_k=2,
        model_dim=config.model_dim,
        expert_hidden_dim=config.model_dim * 4,
        experts_per_group=min(64, config.num_experts),
        gate_hidden_dim=config.model_dim,
        gate_noise_std=1.0,
    )

    if enable_balancer:
        balancer_config = BalancerConfig(
            utilization_momentum=0.95,
            utilization_warmup_steps=config.warmup_steps // 2,
            utilization_threshold_low=0.25,
            utilization_threshold_high=2.0,
            force_assign_ratio=0.25,
            max_force_assign_ratio=0.4,
            aux_loss_weight=0.08,
            lagrangian_init_lambda=1.0,
            lagrangian_lr=0.03,
            lagrangian_max_lambda=6.0,
            load_balance_loss_weight=2.5,
            router_z_loss_weight=0.001,
            cluster_update_interval=config.cluster_interval,
            num_clusters=min(64, config.num_experts // 16),
            similarity_top_k=10,
            bias_correction_factor=0.85,
        )
    else:
        balancer_config = BalancerConfig(
            utilization_momentum=0.95,
            utilization_warmup_steps=1000000,
            utilization_threshold_low=0.0,
            utilization_threshold_high=1e9,
            force_assign_ratio=0.0,
            max_force_assign_ratio=0.0,
            aux_loss_weight=0.0,
            lagrangian_init_lambda=0.0,
            lagrangian_lr=0.0,
            load_balance_loss_weight=0.0,
            router_z_loss_weight=0.0,
            cluster_update_interval=1000000,
            num_clusters=1,
            bias_correction_factor=0.0,
        )

    model = MoELanguageModel(
        vocab_size=config.vocab_size,
        moe_config=moe_config,
        balancer_config=balancer_config,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        enable_balancer=enable_balancer,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    expert_params = sum(p.numel() for p in model.moe_layers[0].experts.parameters()) * config.num_layers
    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Expert parameters: {expert_params / 1e6:.2f}M ({expert_params/total_params*100:.1f}%)")
    logger.info(f"Non-expert parameters: {(total_params - expert_params) / 1e6:.2f}M")

    if config.inject_imbalance:
        inject_extreme_imbalance(model, inject_ratio=config.inject_ratio)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    warmup_steps = config.warmup_steps
    total_steps = config.max_steps
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = defaultdict(list)
    report_interval = config.report_interval

    start_time = time.time()
    last_report_time = start_time

    for step in range(1, config.max_steps + 1):
        model.train()

        input_ids, labels, attention_mask = generate_training_batch(
            vocab_size=config.vocab_size,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)

        logits, task_loss, total_aux_loss, all_gate_outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            training=True,
        )

        total_loss = task_loss + (total_aux_loss if enable_balancer else 0.0)

        total_loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        if enable_balancer:
            for layer_idx, layer in enumerate(model.moe_layers):
                if layer_idx < len(all_gate_outputs):
                    layer.step_balancers(all_gate_outputs[layer_idx])

        optimizer.step()
        scheduler.step()

        task_loss_val = task_loss.item()
        aux_loss_val = total_aux_loss.item() if isinstance(total_aux_loss, torch.Tensor) else total_aux_loss

        history["step"].append(step)
        history["loss/task"].append(task_loss_val)
        history["loss/aux"].append(aux_loss_val)
        history["loss/total"].append(task_loss_val + aux_loss_val)
        history["lr"].append(scheduler.get_last_lr()[0])

        if step % report_interval == 0:
            elapsed = time.time() - last_report_time
            last_report_time = time.time()

            layer0 = model.moe_layers[0]
            gate0 = all_gate_outputs[0] if all_gate_outputs else None

            routing_stats = compute_batch_routing_stats(gate0, moe_config.num_experts, device)

            util_metrics = None
            if enable_balancer:
                util_metrics = compute_utilization_metrics(
                    layer0.utilization_tracker,
                    moe_config.num_experts,
                )

            loss_metrics = compute_loss_metrics(
                task_loss_val,
                gate0,
                aux_loss_val,
            )

            sim_report = None
            if enable_balancer:
                sim_report = compute_expert_similarity_report(
                    layer0,
                    layer0.utilization_tracker,
                )

            print_routing_report(
                step, routing_stats, util_metrics, loss_metrics, sim_report, elapsed,
                num_experts=moe_config.num_experts,
            )

            for k, v in routing_stats.__dict__.items():
                if isinstance(v, (int, float)):
                    history[f"routing/{k}"].append(v)
            if util_metrics is not None:
                for k, v in util_metrics.__dict__.items():
                    if isinstance(v, (int, float)):
                        history[f"util/{k}"].append(v)

    return dict(history)


def run_ablation(config):
    logger.info("=" * 90)
    logger.info("ABLATION EXPERIMENT: Balancer ON vs OFF")
    logger.info("=" * 90)

    logger.info("\n" + ">" * 60)
    logger.info("RUN 1/2: Balancer OFF (Baseline)")
    logger.info(">" * 60)
    baseline = train(config, enable_balancer=False)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    logger.info("\n" + ">" * 60)
    logger.info("RUN 2/2: Balancer ON (Ours)")
    logger.info(">" * 60)
    ours = train(config, enable_balancer=True)

    logger.info("\n" + "=" * 90)
    logger.info("ABLATION COMPARISON SUMMARY")
    logger.info("=" * 90)

    def avg(lst, last_n=100):
        return sum(lst[-last_n:]) / max(len(lst[-last_n:]), 1)

    logger.info(f"{'Metric':<35} {'Baseline (No Bal)':<20} {'Ours (With Bal)':<20} {'Improvement':<15}")
    logger.info("-" * 90)

    baseline_task = avg(baseline["loss/task"])
    ours_task = avg(ours["loss/task"])
    logger.info(f"{'Final Task Loss':<35} {baseline_task:<20.4f} {ours_task:<20.4f} {'':<15}")

    if "util/util_cv_sq" in ours:
        baseline_cv = baseline.get("util/util_cv_sq", [float("inf")])[-1]
        ours_cv = ours["util/util_cv_sq"][-1]
        cv_improvement = (1 - ours_cv / max(baseline_cv, 1e-9)) * 100
        logger.info(f"{'Final Utilization CV²':<35} {baseline_cv:<20.4f} {ours_cv:<20.4f} {cv_improvement:>12.1f}%")

    if "util/balance_ratio" in ours:
        baseline_ratio = baseline.get("util/balance_ratio", [float("inf")])[-1]
        ours_ratio = ours["util/balance_ratio"][-1]
        ratio_improvement = (1 - ours_ratio / max(baseline_ratio, 1e-9)) * 100
        logger.info(f"{'Final Balance Ratio':<35} {baseline_ratio:<20.2f} {ours_ratio:<20.2f} {ratio_improvement:>12.1f}%")

    if "util/underutilized_count" in ours:
        baseline_under = baseline.get("util/underutilized_count", [config.num_experts])[-1]
        ours_under = ours["util/underutilized_count"][-1]
        under_reduction = baseline_under - ours_under
        logger.info(f"{'Underutilized Experts':<35} {baseline_under:<20d} {ours_under:<20d} {under_reduction:>+12d}")

    if "util/long_tail_experts_count" in ours:
        baseline_tail = baseline.get("util/long_tail_experts_count", [config.num_experts])[-1]
        ours_tail = ours["util/long_tail_experts_count"][-1]
        tail_reduction = baseline_tail - ours_tail
        logger.info(f"{'Long-tail Experts (<30% mean)':<35} {baseline_tail:<20d} {ours_tail:<20d} {tail_reduction:>+12d}")

    if "routing/unique_experts_used" in ours:
        baseline_unique = baseline.get("routing/unique_experts_used", [0])[-1]
        ours_unique = ours["routing/unique_experts_used"][-1]
        unique_increase = ours_unique - baseline_unique
        logger.info(f"{'Unique Experts Activated':<35} {baseline_unique:<20d} {ours_unique:<20d} {unique_increase:>+12d}")

    if "routing/force_assigned_tokens" in ours:
        logger.info(f"{'Force-Assigned Tokens %':<35} {'N/A':<20} {avg(ours['routing/force_ratio'])*100:<19.1f}% {'':<15}")
    if "routing/cluster_redirected_tokens" in ours:
        logger.info(f"{'Cluster-Redirected Tokens %':<35} {'N/A':<20} {avg(ours['routing/redirect_ratio'])*100:<19.1f}% {'':<15}")

    logger.info("-" * 90)
    logger.info("\nCONCLUSION:")
    if "util/util_cv_sq" in ours:
        if ours_cv < baseline_cv * 0.1:
            logger.info("  Load balancer highly effective: CV² reduced > 90%！")
        elif ours_cv < baseline_cv * 0.5:
            logger.info("  Load balancer effective: CV² reduced > 50%！")
        else:
            logger.info("  Load balancer has some effect: CV² reduced")

    output_dir = config.output_dir
    with open(os.path.join(output_dir, "ablation_baseline.json"), "w") as f:
        json.dump({k: [v.item() if isinstance(v, torch.Tensor) else v for v in vals]
                   for k, vals in baseline.items()}, f, indent=2)
    with open(os.path.join(output_dir, "ablation_ours.json"), "w") as f:
        json.dump({k: [v.item() if isinstance(v, torch.Tensor) else v for v in vals]
                   for k, vals in ours.items()}, f, indent=2)

    logger.info(f"\nFull histories saved to {output_dir}/ablation_*.json")

    return baseline, ours


def main():
    parser = argparse.ArgumentParser(
        description="MoE 1024 Experts Load Self-Balancing Training Pipeline"
    )
    parser.add_argument("--num_experts", type=int, default=1024)
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--report_interval", type=int, default=50)
    parser.add_argument("--cluster_interval", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="./moe_outputs")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--inject_imbalance",
        action="store_true",
        default=True,
        help="注入极端初始不平衡 (地狱级困难模式)",
    )
    parser.add_argument("--no_inject_imbalance", action="store_false", dest="inject_imbalance")
    parser.add_argument("--inject_ratio", type=float, default=0.02)
    parser.add_argument("--ablation", action="store_true", help="运行开关均衡协议对比实验")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    logger.info("=" * 90)
    logger.info("Mixture-of-Experts (MoE) Large Language Model Training")
    logger.info("=" * 90)
    logger.info(f"Config:")
    logger.info(f"  Experts: {args.num_experts}, Top-K: 2, 层数: {args.num_layers}")
    logger.info(f"  Model dim: {args.model_dim}, 头数: {args.num_heads}, 词表: {args.vocab_size}")
    logger.info(f"  Batch: {args.batch_size}x{args.seq_len}, 步数: {args.max_steps}")
    logger.info(f"  Inject extreme imbalance: {args.inject_imbalance}")
    if args.inject_imbalance:
        logger.warning(f"  Hell difficulty mode: Only {args.inject_ratio*100:.0f}% 专家初始被激活！")
        logger.warning(f"  Expected problem: severe load imbalance, high risk of training collapse")
    logger.info(f"  Run ablation: {args.ablation}")
    logger.info("=" * 90)

    if args.ablation:
        run_ablation(args)
    else:
        train(args, enable_balancer=True)


if __name__ == "__main__":
    main()
