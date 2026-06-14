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
print(f"num_hot experts: {num_hot}")
with torch.no_grad():
    final_layer = layer.gate.gate_proj[-1]
    hot_idx = torch.randperm(moe_cfg.num_experts)[:num_hot]
    print(f"hot_idx: {hot_idx.tolist()}")
    bias = torch.zeros(moe_cfg.num_experts)
    bias[hot_idx] = 8.0
    others_mask = ~torch.isin(torch.arange(moe_cfg.num_experts), hot_idx)
    bias[others_mask] = -2.0
    final_layer.bias.add_(bias)
    print(f"bias added, top5: {final_layer.bias.topk(5)[0].tolist()}")

init_cv_sq = None
final_cv_sq = None
cv_history = []

optimizer = torch.optim.Adam(layer.parameters(), lr=5e-3)

for step in range(100):
    optimizer.zero_grad(set_to_none=True)
    x = torch.randn(4, 16, 32)
    target = torch.randn_like(x)

    out = layer(x, training=True)
    task_loss = F.mse_loss(out.hidden_states, target)
    total_loss = task_loss + out.total_aux_loss

    total_loss.backward()
    optimizer.step()

    if "util/cv_sq" in out.metrics:
        cv_history.append(out.metrics["util/cv_sq"])
        if init_cv_sq is None and step > 10:
            init_cv_sq = out.metrics["util/cv_sq"]
        if step >= 90:
            final_cv_sq = out.metrics["util/cv_sq"]

    if step % 10 == 0:
        fr = out.gate_output.routing_stats.get("force_assign_ratio", 0)
        lb = out.metrics.get("util/cv_sq", float("nan"))
        lo = out.total_aux_loss.item()
        print(f"Step {step:3d} | task={task_loss.item():.4f} aux={lo:.4f} | cv2={lb:.4f} force={fr:.3f}")

print(f"\n=== Summary ===")
print(f"CV history length: {len(cv_history)}")
print(f"Initial CV²: {init_cv_sq:.6f}" if init_cv_sq else "Initial CV²: N/A")
print(f"Final CV²: {final_cv_sq:.6f}" if final_cv_sq else "Final CV²: N/A")
if init_cv_sq and final_cv_sq:
    print(f"CV² reduction: {(1 - final_cv_sq/init_cv_sq) * 100:.2f}%")
    print(f"Improved 50%? {final_cv_sq < init_cv_sq * 0.5}")
