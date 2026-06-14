import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F

from moe_balancer import (
    MoEConfig, BalancerConfig,
    DistributedMoELayer,
)

torch.manual_seed(42)
torch.autograd.set_detect_anomaly(True)

moe_cfg = MoEConfig(num_experts=64, top_k=2, model_dim=32, expert_hidden_dim=64, experts_per_group=16)
bal_cfg = BalancerConfig(
    utilization_warmup_steps=0,
    force_assign_ratio=0.4,
    utilization_threshold_low=0.1,
    aux_loss_weight=0.1,
    cluster_update_interval=3,
    num_clusters=4,
)
layer = DistributedMoELayer(moe_cfg, bal_cfg)

with torch.no_grad():
    final = layer.gate.gate_proj[-1]
    bias = torch.zeros(64)
    bias[:2] = 10.0
    bias[2:] = -3.0
    final.bias.add_(bias)

optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

for step in range(10):
    print(f"\n=== Step {step} ===")
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(4, 16, 32)
    target = torch.randn_like(x)

    out = layer(x, training=True)
    print(f"  forward done, gate_out.aux_loss requires_grad: {out.gate_output.aux_loss.requires_grad}")
    print(f"  total_aux_loss requires_grad: {out.total_aux_loss.requires_grad}")
    print(f"  hidden_states requires_grad: {out.hidden_states.requires_grad}")

    task = F.mse_loss(out.hidden_states, target)
    print(f"  task requires_grad: {task.requires_grad}")
    total = task + out.total_aux_loss
    print(f"  total requires_grad: {total.requires_grad}, grad_fn: {total.grad_fn}")

    try:
        total.backward()
        print("  BACKWARD SUCCESS")
    except RuntimeError as e:
        print(f"  BACKWARD FAILED: {e}")
        break

    optimizer.step()
    print(f"  optimizer step done")
