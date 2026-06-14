replacements = [
    ("--- 路由统计 (Routing Stats) ---", "--- Routing Stats ---"),
    ("  Total tokens:", "  Total tokens:"),
    ("  Normal top-2 命中:", "  Normal top-2 routing:"),
    ("  强制分配 (均衡触发):", "  Forced assignment (balancing):"),
    ("  聚类重分配 (均衡触发):", "  Cluster redirect (balancing):"),
    ("  激活专家数:", "  Unique experts used:"),
    ("  Top1 专家份额:", "  Top1 expert share:"),
    ("  Top5 专家份额:", "  Top5 expert share:"),
    ("  路由熵:", "  Routing entropy:"),
    ("--- 专家利用率 (Utilization) ---", "--- Expert Utilization ---"),
    ("  Max:", "  Max:"),
    ("  Min:", "  Min:"),
    ("  Mean:", "  Mean:"),
    ("  Std:", "  Std:"),
    ("  CV²:", "  CV²:"),
    ("  <-- 越小越均衡，理想 < 0.1", "  <-- smaller = better, ideal < 0.1"),
    ("  负载比:", "  Balance ratio:"),
    ("  <-- Max/Min，理想 < 1.5", "  <-- Max/Min, ideal < 1.5"),
    ("  欠载专家:", "  Underutilized:"),
    ("  过载专家:", "  Overutilized:"),
    ("  长尾专家:", "  Long-tail experts:"),
    ("  (< 30% 均值)", "  (< 30% of mean)"),
    ("--- 损失 (Losses) ---", "--- Losses ---"),
    ("  主任务 LM Loss:", "  Task LM Loss:"),
    ("  负载均衡损失:", "  Load balance loss:"),
    ("  拉格朗日增广损失:", "  Augmented Lagrangian:"),
    ("  梯度偏差校正:", "  Gradient bias correction:"),
    ("  总辅助损失:", "  Total aux loss:"),
    ("  总损失:", "  Total loss:"),
    ("--- 专家相似度与聚类 ---", "--- Expert Similarity & Clustering ---"),
    ("  聚类数:", "  Num clusters:"),
    ("  聚类大小标准差:", "  Cluster size std:"),
    ("  相似度均值/最大/最小:", "  Similarity mean/max/min:"),
    ("  Top 相似专家对:", "  Top similar expert pairs:"),
    ("  欠载专家聚类分布:", "  Underutilized per cluster:"),
    ("  欠载专家:", "  Underutilized:"),
    ("ABLATION EXPERIMENT: 开启均衡协议 vs 关闭均衡协议", "ABLATION EXPERIMENT: Balancer ON vs OFF"),
    ("RUN 1/2: 关闭均衡协议 (Baseline)", "RUN 1/2: Balancer OFF (Baseline)"),
    ("RUN 2/2: 开启均衡协议 (Ours)", "RUN 2/2: Balancer ON (Ours)"),
    ("ABLATION COMPARISON SUMMARY", "ABLATION COMPARISON SUMMARY"),
    ("Final Task Loss", "Final Task Loss"),
    ("Final Utilization CV²", "Final Utilization CV²"),
    ("Final Balance Ratio", "Final Balance Ratio"),
    ("Underutilized Experts", "Underutilized Experts"),
    ("Long-tail Experts (<30% mean)", "Long-tail Experts (<30% mean)"),
    ("Unique Experts Activated", "Unique Experts Activated"),
    ("Force-Assigned Tokens %", "Force-Assigned Tokens %"),
    ("Cluster-Redirected Tokens %", "Cluster-Redirected Tokens %"),
    ("Improvement", "Improvement"),
    ("CONCLUSION:", "CONCLUSION:"),
    ("  负载均衡协议效果显著: CV² 降低 > 90%！", "  Load balancer highly effective: CV² reduced > 90%！"),
    ("  负载均衡协议效果良好: CV² 降低 > 50%！", "  Load balancer effective: CV² reduced > 50%！"),
    ("  负载均衡协议有一定效果: CV² 有所降低", "  Load balancer has some effect: CV² reduced"),
    ("Mixture-of-Experts (MoE) 大规模语言模型训练", "Mixture-of-Experts (MoE) Large Language Model Training"),
    ("配置:", "Config:"),
    ("  专家数:", "  Experts:"),
    ("  Top-K: 2, 层数:", "  Top-K: 2, Layers:"),
    ("  模型维度:", "  Model dim:"),
    ("  头数:", "  Heads:"),
    ("  词表:", "  Vocab:"),
    ("  Batch:", "  Batch:"),
    ("  步数:", "  Steps:"),
    ("  注入极端不平衡:", "  Inject extreme imbalance:"),
    ("  地狱级困难模式: 仅", "  Hell difficulty mode: Only"),
    ("  专家初始被激活！", "  experts will be initially favored！"),
    ("  预期问题: 专家负载极端不均，训练崩溃风险高", "  Expected problem: severe load imbalance, high risk of training collapse"),
    ("  运行消融对比:", "  Run ablation:"),
    ("Injecting extreme imbalance: Only", "Injecting extreme imbalance: Only"),
    ("experts will be initially favored (地狱级困难 scenario)", "experts will be initially favored (hell difficulty scenario)"),
    ("MoE 1024 Experts 负载自平衡训练主流程", "MoE 1024 Experts Load Self-Balancing Training Pipeline"),
    ("MoE Load Self-Balancing Training - Real 1024 Expert Routing", "MoE Load Self-Balancing Training - Real 1024 Expert Routing"),
    ("  Layer", "  Layer"),
    ("Hot experts =", "Hot experts ="),
    ("  (/ 1024)", "  (/ 1024)"),
    ("(/ 1024)  (< 30% mean)", "(/ 1024)  (< 30% mean)"),
]

with open("d:/Worksolo/s0003/train_1024expert.py", "r", encoding="utf-8") as f:
    content = f.read()

for old, new in replacements:
    content = content.replace(old, new)

# Also fix the hardcoded 1024 values
content = content.replace("  / 1024", "  / {config.num_experts}")

# Fix long hardcoded 1024 in print_routing_report
import re
content = content.replace("  / 1024\n", "  / {config.num_experts}\n")

# Also handle the print_routing_report that has hardcoded 1024 in the main loop
content = content.replace("  / 1024  /", "  / {config.num_experts}  /")
content = content.replace("  / 1024)", "  / {config.num_experts})")
content = content.replace("  / 1024:", "  / {config.num_experts}:")
content = content.replace("  / 1024", "  / {config.num_experts}")

with open("d:/Worksolo/s0003/train_1024expert.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Done!")
