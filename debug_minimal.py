import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F

from moe_balancer import (
    MoEConfig, BalancerConfig,
    DistributedMoELayer,
    SmartMoEGate,
    MovingAverageUtilizationTracker,
)

torch.autograd.set_detect_anomaly(True)

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

moe_cfg = MoEConfig(
    num_experts=16, top_k=2, model_dim=8, expert_hidden_dim=16, experts_per_group=8
)
bal_cfg = BalancerConfig(
    utilization_momentum=0.9,
    utilization_warmup_steps=2,
    utilization_threshold_low=0.2,
    utilization_threshold_high=2.0,
    force_assign_ratio=0.0,
    aux_loss_weight=0.1,
    lagrangian_lr=0.05,
    load_balance_loss_weight=1.0,
    cluster_update_interval=100,
)

gate = SmartMoEGate(moe_cfg, bal_cfg)
tracker = MovingAverageUtilizationTracker(moe_cfg, bal_cfg)

from moe_balancer import LagrangianBalancer
lagrangian = LagrangianBalancer(moe_cfg.num_experts, bal_cfg)

print("=== Testing with tracker + lagrangian ===")

for step in range(10):
    print(f"\nStep {step}")
    x = torch.randn(2, 4, 8, requires_grad=True)
    out = gate(x, tracker=tracker, clusterer=None, lagrangian=lagrangian, training=True)

    print(f"  tracker.step_count = {tracker.step_count}")
    print(f"  expert_weights requires_grad: {out.expert_weights.requires_grad}")
    print(f"  aux_loss requires_grad: {out.aux_loss.requires_grad}")
    print(f"  aux_loss = {out.aux_loss.item():.4f}")
    print(f"  lambda_plus[0] = {lagrangian._lambda_plus[0].item():.4f}")

    loss = out.expert_weights.sum() + out.aux_loss
    print(f"  loss = {loss.item():.4f}, requires_grad = {loss.requires_grad}")

    try:
        loss.backward()
        print("  BACKWARD OK")
    except RuntimeError as e:
        print(f"  BACKWARD FAILED: {e}")
        break

    x.grad = None
    for p in gate.parameters():
        if p.grad is not None:
            p.grad = None
