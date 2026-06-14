import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F

from moe_balancer import (
    MoEConfig, BalancerConfig,
    DistributedMoELayer,
)

torch.autograd.set_detect_anomaly(True)

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

moe_cfg = MoEConfig(
    num_experts=64, top_k=2, model_dim=32, expert_hidden_dim=64, experts_per_group=16
)
bal_cfg = BalancerConfig(
    utilization_momentum=0.9,
    utilization_warmup_steps=10,
    utilization_threshold_low=0.2,
    utilization_threshold_high=2.0,
    force_assign_ratio=0.3,
    aux_loss_weight=0.1,
    lagrangian_lr=0.05,
    load_balance_loss_weight=3.0,
    cluster_update_interval=20,
    num_clusters=8,
)
layer = DistributedMoELayer(moe_cfg, bal_cfg)

num_hot = max(1, int(moe_cfg.num_experts * 0.03))
with torch.no_grad():
    final_layer = layer.gate.gate_proj[-1]
    hot_idx = torch.randperm(moe_cfg.num_experts)[:num_hot]
    bias = torch.zeros(moe_cfg.num_experts)
    bias[hot_idx] = 8.0
    others_mask = ~torch.isin(torch.arange(moe_cfg.num_experts), hot_idx)
    bias[others_mask] = -2.0
    final_layer.bias.add_(bias)

optimizer = torch.optim.Adam(layer.parameters(), lr=5e-3)

for step in range(20):
    print(f"\n=== Step {step} ===")
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(4, 16, 32)
    target = torch.randn_like(x)

    out = layer(x, training=True)
    print(f"  forward done")
    print(f"  aux_loss: {out.total_aux_loss.item():.4f}, requires_grad={out.total_aux_loss.requires_grad}")
    print(f"  gate aux_loss: {out.gate_output.aux_loss.item():.4f}, requires_grad={out.gate_output.aux_loss.requires_grad}")

    task_loss = F.mse_loss(out.hidden_states, target)
    total_loss = task_loss + out.total_aux_loss
    print(f"  total_loss: {total_loss.item():.4f}, requires_grad={total_loss.requires_grad}")

    try:
        total_loss.backward()
        print(f"  backward done")
    except RuntimeError as e:
        print(f"  BACKWARD FAILED: {e}")
        break

    optimizer.step()
    print(f"  step done")

    # Print lambda changes
    lp = layer.lagrangian_balancer._lambda_plus
    lm = layer.lagrangian_balancer._lambda_minus
    print(f"  λ+ mean={lp.mean().item():.4f} max={lp.max().item():.4f}")
    print(f"  λ- mean={lm.mean().item():.4f} max={lm.max().item():.4f}")
