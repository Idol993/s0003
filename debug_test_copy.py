import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F

from moe_balancer import (
    MoEConfig, BalancerConfig,
    DistributedMoELayer,
)

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

moe_cfg = MoEConfig(num_experts=64, top_k=2, model_dim=32, expert_hidden_dim=64)
bal_cfg = BalancerConfig(
    utilization_warmup_steps=0,
    force_assign_ratio=0.4,
    max_force_assign_ratio=0.5,
    utilization_threshold_low=0.1,
    aux_loss_weight=0.1,
    bias_correction_factor=0.8,
)
layer = DistributedMoELayer(moe_cfg, bal_cfg)

print(f"Layer created")
print(f"  gate params: {sum(p.numel() for p in layer.gate.parameters())}")
print(f"  experts params: {sum(p.numel() for p in layer.experts.parameters())}")
print(f"  lagrangian buffers: {sum(b.numel() for b in layer.lagrangian_balancer.buffers())}")

with torch.no_grad():
    final = layer.gate.gate_proj[-1]
    bias = torch.zeros(64)
    bias[:2] = 10.0
    bias[2:] = -3.0
    final.bias.add_(bias)
    print(f"Bias injected, final.bias[:3] = {final.bias[:3]}")

optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

for step in range(5):
    print(f"\n=== Step {step} ===")
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(4, 16, 32)
    target = torch.randn_like(x)

    out = layer(x, training=True)
    task = F.mse_loss(out.hidden_states, target)
    total = task + out.total_aux_loss
    print(f"  task={task.item():.4f}, aux={out.total_aux_loss.item():.4f}, total={total.item():.4f}")
    print(f"  total.requires_grad: {total.requires_grad}")
    print(f"  total.grad_fn: {total.grad_fn}")

    try:
        total.backward()
        print("  BACKWARD SUCCESS")
    except RuntimeError as e:
        print(f"  BACKWARD FAILED: {e}")
        break

    optimizer.step()
    print("  optimizer step done")
