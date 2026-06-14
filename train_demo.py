import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List

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


class SimpleMoELanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        moe_config: MoEConfig,
        balancer_config: BalancerConfig,
        num_layers: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = moe_config.model_dim
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, self.model_dim)
        self.position_embedding = nn.Embedding(1024, self.model_dim)

        self.moe_layers = nn.ModuleList([
            DistributedMoELayer(moe_config, balancer_config)
            for _ in range(num_layers)
        ])

        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(self.model_dim) for _ in range(num_layers)
        ])
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.model_dim,
                num_heads=num_heads,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(self.model_dim)
        self.lm_head = nn.Linear(self.model_dim, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        training: bool = True,
    ):
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

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
                is_causal=False,
            )
            x = residual + attn_out

            moe_out = self.moe_layers[layer_idx](x, attention_mask=attention_mask, training=training)
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


def generate_biased_data(
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    bias_strength: float = 0.8,
    num_bias_tokens: int = 10,
):
    probs = torch.ones(vocab_size) / vocab_size
    bias_idx = torch.randperm(vocab_size)[:num_bias_tokens]
    bias_val = bias_strength / num_bias_tokens
    probs[:] = (1.0 - bias_strength) / (vocab_size - num_bias_tokens)
    probs[bias_idx] = bias_val

    input_ids = torch.multinomial(probs, batch_size * seq_len, replacement=True)
    input_ids = input_ids.view(batch_size, seq_len)

    labels = input_ids.clone()
    mask = torch.rand_like(labels.float()) > 0.15
    labels[mask] = -100

    attention_mask = torch.ones(batch_size, seq_len)
    return input_ids.long(), labels.long(), attention_mask.long()


def inject_extreme_imbalance(moe_layer: DistributedMoELayer, inject_ratio: float = 0.05):
    num_hot = max(1, int(moe_layer.num_experts * inject_ratio))
    logger.warning(
        f"Injecting extreme imbalance: favoring {num_hot}/{moe_layer.num_experts} experts "
        f"(simulating 地狱级困难 scenario)"
    )

    with torch.no_grad():
        gate_proj_out = moe_layer.gate.gate_proj[-1]
        hot_experts = torch.randperm(moe_layer.num_experts)[:num_hot]
        cold_experts = [i for i in range(moe_layer.num_experts) if i not in hot_experts.tolist()]

        bias_add = torch.zeros(moe_layer.num_experts)
        bias_add[hot_experts] = 5.0
        bias_add[cold_experts] = -2.0

        gate_proj_out.bias.add_(bias_add.to(gate_proj_out.bias.device))


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() and not config.cpu else "cpu")
    logger.info(f"Using device: {device}")

    moe_config = MoEConfig(
        num_experts=config.num_experts,
        top_k=2,
        model_dim=config.model_dim,
        expert_hidden_dim=config.model_dim * 4,
        experts_per_group=min(64, config.num_experts),
    )

    balancer_config = BalancerConfig(
        utilization_momentum=0.95,
        utilization_warmup_steps=50,
        utilization_threshold_low=0.2,
        utilization_threshold_high=2.5,
        force_assign_ratio=0.2,
        aux_loss_weight=0.05,
        lagrangian_init_lambda=1.0,
        lagrangian_lr=0.02,
        lagrangian_max_lambda=8.0,
        load_balance_loss_weight=2.0,
        router_z_loss_weight=0.001,
        cluster_update_interval=30,
        num_clusters=min(32, config.num_experts // 4),
    )

    model = SimpleMoELanguageModel(
        vocab_size=config.vocab_size,
        moe_config=moe_config,
        balancer_config=balancer_config,
        num_layers=config.num_layers,
    ).to(device)

    if config.inject_imbalance:
        for layer in model.moe_layers:
            inject_extreme_imbalance(layer, inject_ratio=config.inject_ratio)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Expert params (estimated): {config.num_layers * config.num_experts * (3 * config.model_dim * 4 * config.model_dim) / 1e6:.2f}M")

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
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = defaultdict(list)
    report_interval = config.report_interval

    start_time = time.time()
    best_balance = float("inf")

    for step in range(1, config.max_steps + 1):
        model.train()

        input_ids, labels, attention_mask = generate_biased_data(
            vocab_size=config.vocab_size,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            bias_strength=0.6,
        )
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits, loss, aux_loss, all_gate_outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            training=True,
        )

        total_loss = loss + aux_loss

        total_loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        for layer_idx, layer in enumerate(model.moe_layers):
            if layer_idx < len(all_gate_outputs):
                layer.step_balancers(all_gate_outputs[layer_idx])

        optimizer.step()
        scheduler.step()

        history["loss/train"].append(loss.item())
        history["loss/aux"].append(aux_loss.item() if isinstance(aux_loss, torch.Tensor) else aux_loss)
        history["loss/total"].append(total_loss.item())
        history["lr"].append(scheduler.get_last_lr()[0])

        if step % report_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = report_interval / elapsed
            start_time = time.time()

            avg = lambda k: sum(history[k][-report_interval:]) / max(len(history[k][-report_interval:]), 1)

            last_layer = model.moe_layers[-1]
            report = last_layer.get_balance_report()

            balance_cv = report.get("cv_sq", float("nan"))
            balance_ratio = report.get("balance_ratio", float("nan"))
            num_under = report.get("num_under", -1)
            num_over = report.get("num_over", -1)
            util_std = report.get("util_std", float("nan"))
            util_max = report.get("util_max", float("nan"))
            util_min = report.get("util_min", float("nan"))
            long_tail = report.get("long_tail_count", -1)

            history["util/cv_sq"].append(balance_cv)
            history["util/balance_ratio"].append(balance_ratio)
            history["util/util_max"].append(util_max)
            history["util/util_min"].append(util_min)
            history["util/util_std"].append(util_std)
            history["util/num_under"].append(num_under)
            history["util/num_over"].append(num_over)
            history["util/long_tail"].append(long_tail)

            gate0 = all_gate_outputs[0] if all_gate_outputs else None
            if gate0 is not None:
                force_mask = gate0.force_assign_mask.reshape(-1)
                redirect_mask = gate0.cluster_redirect_mask.reshape(-1)
                total_slots = force_mask.numel()
                force_ratio = force_mask.float().mean().item()
                redirect_ratio = redirect_mask.float().mean().item()
                both_ratio = (force_mask & redirect_mask).float().mean().item()
                normal_ratio = (~(force_mask | redirect_mask)).float().mean().item()
                unique_experts = (torch.bincount(
                    gate0.expert_indices.reshape(-1).cpu(),
                    minlength=moe_config.num_experts
                ) > 0).sum().item()

                history["routing/force_ratio"].append(force_ratio)
                history["routing/redirect_ratio"].append(redirect_ratio)
                history["routing/both_ratio"].append(both_ratio)
                history["routing/normal_ratio"].append(normal_ratio)
                history["routing/unique_experts"].append(unique_experts)
            else:
                force_ratio = float("nan")
                redirect_ratio = float("nan")
                both_ratio = float("nan")
                normal_ratio = float("nan")

            logger.info(
                f"Step {step}/{config.max_steps} | "
                f"Loss: {avg('loss/train'):.4f} "
                f"(aux={avg('loss/aux'):.4f}) | "
                f"CV^2: {balance_cv:.4f} | "
                f"Ratio: {balance_ratio:.2f} | "
                f"Util: {util_max*100:.1f}%/{util_min*100:.1f}% | "
                f"Under: {num_under} Over: {num_over} | "
                f"Rate: {steps_per_sec:.1f} steps/s"
            )
            logger.info(
                f"  Routing: Normal={normal_ratio*100:.1f}% "
                f"Force={force_ratio*100:.1f}% "
                f"Redirect={redirect_ratio*100:.1f}% "
                f"Both={both_ratio*100:.1f}%"
            )

            if balance_cv < best_balance and step > warmup_steps and not math.isnan(balance_cv):
                best_balance = balance_cv
                logger.info(f"  [IMPROVED] Best balance CV^2: {best_balance:.6f}")

            lambdas = report["lambda_stats"]
            logger.info(
                f"  Lambda+: mean={lambdas['plus_mean']:.3f} max={lambdas['plus_max']:.3f} | "
                f"Lambda-: mean={lambdas['minus_mean']:.3f} max={lambdas['minus_max']:.3f}"
            )

            if report.get("num_clusters"):
                logger.info(
                    f"  Clusters: {report['num_clusters']} | "
                    f"Size std: {report['cluster_size_std']:.2f}"
                )

    history_path = os.path.join(config.output_dir, "training_history.json")
    serializable = {}
    for k, v in history.items():
        if len(v) > 0 and isinstance(v[0], (int, float)):
            serializable[k] = v
        elif len(v) > 0 and isinstance(v[0], torch.Tensor):
            serializable[k] = [x.item() if hasattr(x, "item") else float(x) for x in v]
    with open(history_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    final_report = model.moe_layers[-1].get_balance_report()
    report_path = os.path.join(config.output_dir, "final_balance_report.json")
    def to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, list):
            return [to_serializable(x) for x in obj]
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        return obj
    with open(report_path, "w") as f:
        json.dump(to_serializable(final_report), f, indent=2)
    logger.info(f"Final balance report saved to {report_path}")

    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Final Balance CV^2: {history.get('util/cv_sq', [float('nan')])[-1]:.6f}")
    logger.info(f"Final Balance Ratio: {history.get('util/balance_ratio', [float('nan')])[-1]:.2f}")
    logger.info(f"Final Underutilized: {final_report.get('num_under', 'N/A')} / {config.num_experts}")
    logger.info(f"Final Overutilized: {final_report.get('num_over', 'N/A')} / {config.num_experts}")
    logger.info(f"Aux Loss Contribution: {sum(history['loss/aux'][-100:])/max(len(history['loss/aux'][-100:]),1) * 100 / max(sum(history['loss/total'][-100:])/max(len(history['loss/total'][-100:]),1), 1e-9):.2f}%")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="MoE Distributed Load Balancer Training Demo - 地狱级困难负载均衡验证"
    )
    parser.add_argument("--num_experts", type=int, default=1024)
    parser.add_argument("--model_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--report_interval", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument(
        "--inject_imbalance",
        action="store_true",
        default=True,
        help="Inject extreme initial imbalance (地狱级困难)",
    )
    parser.add_argument("--no_inject_imbalance", action="store_false", dest="inject_imbalance")
    parser.add_argument("--inject_ratio", type=float, default=0.02, help="Ratio of hot experts for imbalance injection")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MoE DISTRIBUTED LOAD BALANCER TRAINING DEMO")
    logger.info("=" * 60)
    logger.info(f"Experts: {args.num_experts}, Top-K: 2, Layers: {args.num_layers}")
    logger.info(f"Model Dim: {args.model_dim}, Vocab: {args.vocab_size}")
    logger.info(f"Batch: {args.batch_size}x{args.seq_len}, Steps: {args.max_steps}")
    logger.info(f"Inject Extreme Imbalance: {args.inject_imbalance}")
    if args.inject_imbalance:
        logger.warning("  [WARN] Hell difficulty: Only ~2% experts will be initially activated!")
        logger.warning("  [WARN] Expected initial CV^2 >> 100, should recover to < 0.5 ideally")
    logger.info("=" * 60)

    train(args)


if __name__ == "__main__":
    main()
