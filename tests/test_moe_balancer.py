import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import unittest
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe_balancer import (
    MoEConfig,
    BalancerConfig,
    DistributedConfig,
    ExpertLayer,
    ExpertGroup,
    MovingAverageUtilizationTracker,
    ExpertSimilarityClusterer,
    LagrangianBalancer,
    SmartMoEGate,
    DistributedMoELayer,
    compute_cosine_similarity,
    safe_softmax,
    gumbel_softmax_sample,
)


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TestUtils(unittest.TestCase):
    def test_safe_softmax(self):
        set_seed()
        logits = torch.randn(4, 16) * 100
        out = safe_softmax(logits, dim=-1)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((out > 0).all())
        self.assertTrue(torch.allclose(out.sum(dim=-1), torch.ones(4), atol=1e-5))

    def test_cosine_similarity(self):
        set_seed()
        x = torch.randn(8, 32)
        sim = compute_cosine_similarity(x)
        self.assertEqual(sim.shape, (8, 8))
        self.assertTrue(torch.allclose(torch.diag(sim), torch.ones(8), atol=1e-5))
        self.assertTrue((sim >= -1 - 1e-5).all() and (sim <= 1 + 1e-5).all())

    def test_gumbel_softmax(self):
        set_seed()
        logits = torch.randn(16, 32)
        out_soft = gumbel_softmax_sample(logits, temperature=1.0, hard=False)
        self.assertTrue(torch.isfinite(out_soft).all())
        self.assertTrue((out_soft > 0).all())
        self.assertTrue(torch.allclose(out_soft.sum(dim=-1), torch.ones(16), atol=1e-5))

        out_hard = gumbel_softmax_sample(logits, temperature=1.0, hard=True)
        self.assertEqual(out_hard.shape, (16, 32))
        self.assertTrue(torch.allclose(out_hard.sum(dim=-1), torch.ones(16), atol=1e-5))


class TestExpertLayer(unittest.TestCase):
    def test_single_expert(self):
        set_seed()
        expert = ExpertLayer(model_dim=64, hidden_dim=256, activation="silu")
        x = torch.randn(8, 16, 64)
        out = expert(x)
        self.assertEqual(out.shape, (8, 16, 64))
        self.assertTrue(torch.isfinite(out).all())

    def test_activation_variants(self):
        set_seed()
        for act in ["silu", "gelu", "relu", "leaky_relu"]:
            expert = ExpertLayer(model_dim=32, hidden_dim=128, activation=act)
            x = torch.randn(2, 4, 32)
            out = expert(x)
            self.assertEqual(out.shape, x.shape, f"Failed for activation {act}")

    def test_gradient_flow(self):
        set_seed()
        expert = ExpertLayer(model_dim=32, hidden_dim=128)
        x = torch.randn(2, 4, 32, requires_grad=True)
        out = expert(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        for p in expert.parameters():
            self.assertIsNotNone(p.grad)


class TestExpertGroup(unittest.TestCase):
    def test_group_forward_small(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=16, top_k=2, model_dim=32, expert_hidden_dim=128, experts_per_group=8)
        group = ExpertGroup(moe_cfg)

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 32)

        num_tokens = batch_size * seq_len
        expert_indices = torch.randint(0, 16, (num_tokens, 2))
        expert_weights = F.softmax(torch.randn(num_tokens, 2), dim=-1)

        expert_indices_3d = expert_indices.view(batch_size, seq_len, 2)
        expert_weights_3d = expert_weights.view(batch_size, seq_len, 2)

        out, _ = group(x, expert_indices_3d, expert_weights_3d)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())

    def test_large_expert_count_params(self):
        moe_cfg = MoEConfig(num_experts=512, top_k=2, model_dim=16, expert_hidden_dim=64, experts_per_group=64)
        group = ExpertGroup(moe_cfg)
        self.assertEqual(group.num_local_experts, 512)
        total = sum(p.numel() for p in group.parameters())
        expected_per_expert = (16 * 64 + 64) + (64 * 16 + 16) + (16 * 64 + 64)
        self.assertEqual(total, expected_per_expert * 512)

    def test_expert_representations(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=8, top_k=2, model_dim=16, expert_hidden_dim=64)
        group = ExpertGroup(moe_cfg)
        reps = group.get_expert_representations()
        self.assertEqual(reps.shape[0], 8)
        self.assertGreater(reps.shape[1], 0)


class TestMovingAverageUtilizationTracker(unittest.TestCase):
    def test_update_tracking(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=16)
        bal_cfg = BalancerConfig(utilization_warmup_steps=5)
        tracker = MovingAverageUtilizationTracker(moe_cfg, bal_cfg)

        for step in range(20):
            weights = torch.rand(64, 16)
            weights = weights / weights.sum(dim=-1, keepdim=True)
            tracker.update(weights)

        self.assertEqual(tracker.step_count, 20)
        stats = tracker.get_stats()
        self.assertEqual(stats.moving_avg_utilization.shape, (16,))
        self.assertTrue(torch.isfinite(stats.moving_avg_utilization).all())
        self.assertGreater(stats.total_tokens_processed, 0)

    def test_extreme_imbalance_detection(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=32)
        bal_cfg = BalancerConfig(
            utilization_threshold_low=0.1,
            utilization_threshold_high=3.0,
            utilization_warmup_steps=0,
        )
        tracker = MovingAverageUtilizationTracker(moe_cfg, bal_cfg)

        for step in range(50):
            weights = torch.zeros(128, 32)
            weights[:, 0] = 0.9
            weights[:, 1] = 0.1
            tracker.update(weights)

        stats = tracker.get_stats()
        under = stats.low_utilization_experts
        over = stats.high_utilization_experts

        self.assertGreaterEqual(len(under), 28)
        self.assertGreaterEqual(len(over), 1)
        self.assertGreater(stats.balance_ratio, 10.0)
        self.assertGreater(stats.utilization_cv, 1.0)

    def test_priority_scores(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=16)
        bal_cfg = BalancerConfig(utilization_warmup_steps=0)
        tracker = MovingAverageUtilizationTracker(moe_cfg, bal_cfg)

        for step in range(20):
            weights = torch.zeros(100, 16)
            weights[:, 0] = 0.95
            weights[:, 1] = 0.05
            tracker.update(weights)

        scores = tracker.get_utilization_priority_scores()
        self.assertGreater(scores[2:].mean().item(), scores[0].item())
        self.assertTrue(torch.isfinite(scores).all())

    def test_load_imbalance_penalty(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=16)
        bal_cfg = BalancerConfig(utilization_warmup_steps=0)
        tracker = MovingAverageUtilizationTracker(moe_cfg, bal_cfg)

        for step in range(10):
            weights = torch.zeros(100, 16)
            weights[:, 0] = 1.0
            tracker.update(weights)

        penalty = tracker.compute_load_imbalance_penalty()
        self.assertGreater(penalty.item(), 1.0)

    def test_state_dict_save_load(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=8)
        bal_cfg = BalancerConfig()
        tracker = MovingAverageUtilizationTracker(moe_cfg, bal_cfg)

        for _ in range(15):
            tracker.update(torch.rand(50, 8))

        state = tracker.state_dict()
        tracker2 = MovingAverageUtilizationTracker(moe_cfg, bal_cfg)
        tracker2.load_state_dict(state)

        self.assertEqual(tracker2.step_count, 15)
        self.assertTrue(torch.allclose(
            tracker.get_bias_corrected_utilization(),
            tracker2.get_bias_corrected_utilization(),
            atol=1e-7,
        ))


class TestExpertSimilarityClusterer(unittest.TestCase):
    def test_similarity_matrix(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=16)
        bal_cfg = BalancerConfig(num_clusters=4)
        clusterer = ExpertSimilarityClusterer(moe_cfg, bal_cfg)

        reps = torch.randn(16, 128)
        sim = clusterer.compute_similarity_from_parameters(reps)
        self.assertEqual(sim.shape, (16, 16))
        self.assertTrue(torch.allclose(torch.diag(sim), torch.zeros(16), atol=1e-5))

    def test_kmeans_clustering(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=32)
        bal_cfg = BalancerConfig(num_clusters=8)
        clusterer = ExpertSimilarityClusterer(moe_cfg, bal_cfg)

        reps = torch.randn(32, 256)
        for c in range(8):
            reps[c * 4:(c + 1) * 4] += torch.randn(1, 256) * 5

        clusterer.compute_similarity_from_parameters(reps)
        result = clusterer.cluster_experts_kmeans(max_iter=50)

        self.assertEqual(result.cluster_assignments.shape, (32,))
        self.assertEqual(result.num_clusters, 8)
        self.assertEqual(len(result.cluster_centers), 8)
        total = sum(len(c) for c in result.cluster_centers)
        self.assertEqual(total, 32)

    def test_top_k_similar(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=16)
        bal_cfg = BalancerConfig(similarity_top_k=3)
        clusterer = ExpertSimilarityClusterer(moe_cfg, bal_cfg)

        reps = torch.randn(16, 64)
        clusterer.compute_similarity_from_parameters(reps)
        top_k_idx, top_k_vals = clusterer.get_top_k_similar(0, k=3)
        self.assertEqual(len(top_k_idx), 3)
        self.assertEqual(len(top_k_vals), 3)
        self.assertNotIn(0, top_k_idx.tolist())

    def test_redistribution_mapping(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=32)
        bal_cfg = BalancerConfig(num_clusters=4, similarity_top_k=3)
        clusterer = ExpertSimilarityClusterer(moe_cfg, bal_cfg)

        reps = torch.randn(32, 64)
        clusterer.compute_similarity_from_parameters(reps)
        clusterer.cluster_experts_kmeans()

        over = torch.tensor([0, 1, 2])
        under = torch.tensor([28, 29, 30, 31])
        mapping = clusterer.generate_redistribution_mapping(over, under)
        self.assertIsInstance(mapping, dict)
        for under_idx, src_list in mapping.items():
            self.assertIn(under_idx, [28, 29, 30, 31])
            for s in src_list:
                self.assertIn(s, [0, 1, 2])


class TestLagrangianBalancer(unittest.TestCase):
    def test_constraint_violations(self):
        set_seed()
        bal_cfg = BalancerConfig(
            utilization_threshold_low=0.5,
            utilization_threshold_high=1.5,
        )
        lb = LagrangianBalancer(num_experts=8, balancer_config=bal_cfg)

        perfect_util = torch.full((8,), 1.0 / 8)
        vp, vm = lb.compute_constraint_violations(perfect_util)
        self.assertTrue(torch.allclose(vp, torch.zeros(8), atol=1e-5))
        self.assertTrue(torch.allclose(vm, torch.zeros(8), atol=1e-5))

        bad_util = torch.zeros(8)
        bad_util[0] = 1.0
        vp, vm = lb.compute_constraint_violations(bad_util)
        self.assertGreater(vp[1:].sum().item(), 0)
        self.assertGreater(vm[0].item(), 0)

    def test_multiplier_update(self):
        set_seed()
        bal_cfg = BalancerConfig(
            lagrangian_lr=0.1,
            lagrangian_momentum=0.0,
            utilization_threshold_low=0.5,
            utilization_threshold_high=1.5,
        )
        lb = LagrangianBalancer(num_experts=8, balancer_config=bal_cfg)

        init_lp = lb._lambda_plus.clone()
        init_lm = lb._lambda_minus.clone()

        for _ in range(10):
            bad_util = torch.zeros(8)
            bad_util[0] = 1.0
            info = lb.step_multipliers(bad_util)

        self.assertGreater((lb._lambda_plus[1:] - init_lp[1:]).mean().item(), 0)
        self.assertGreater((lb._lambda_minus[0] - init_lm[0]).item(), 0)
        self.assertTrue((lb._lambda_plus >= bal_cfg.lagrangian_min_lambda).all())
        self.assertTrue((lb._lambda_plus <= bal_cfg.lagrangian_max_lambda).all())

    def test_penalty_computation(self):
        set_seed()
        bal_cfg = BalancerConfig(
            utilization_threshold_low=0.3,
            utilization_threshold_high=2.0,
        )
        lb = LagrangianBalancer(num_experts=8, balancer_config=bal_cfg)

        logits_bad = torch.zeros(64, 8)
        logits_bad[:, 0] = 10.0
        penalty_bad, info_bad = lb.compute_load_balance_penalty(logits_bad)
        self.assertGreater(info_bad["total_penalty"], 0)

        logits_good = torch.randn(64, 8)
        penalty_good, info_good = lb.compute_load_balance_penalty(logits_good)
        self.assertLess(info_good["cv_sq_penalty"], info_bad["cv_sq_penalty"])

    def test_router_z_loss(self):
        set_seed()
        bal_cfg = BalancerConfig()
        lb = LagrangianBalancer(num_experts=16, balancer_config=bal_cfg)

        small_logits = torch.randn(32, 16) * 0.1
        big_logits = torch.randn(32, 16) * 10.0

        z_small = lb.compute_router_z_loss(small_logits)
        z_big = lb.compute_router_z_loss(big_logits)
        self.assertLess(z_small.item(), z_big.item())

    def test_bias_correction(self):
        set_seed()
        bal_cfg = BalancerConfig()
        lb = LagrangianBalancer(num_experts=16, balancer_config=bal_cfg)

        weights = torch.rand(100, 16)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        mask = torch.zeros(100, 2, dtype=torch.bool)
        correction_zero = lb.compute_implicit_gradient_bias_correction(weights, mask)
        self.assertEqual(correction_zero.item(), 0.0)


class TestSmartMoEGate(unittest.TestCase):
    def test_forward_shapes(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=16, top_k=2, model_dim=32, gate_hidden_dim=64)
        bal_cfg = BalancerConfig(force_assign_ratio=0.0)
        gate = SmartMoEGate(moe_cfg, bal_cfg)

        x = torch.randn(2, 8, 32)
        out = gate(x, tracker=None, clusterer=None, lagrangian=None, training=False)

        self.assertEqual(out.expert_indices.shape, (2, 8, 2))
        self.assertEqual(out.expert_weights.shape, (2, 8, 2))
        self.assertEqual(out.raw_logits.shape, (2, 8, 16))
        self.assertTrue(torch.allclose(out.expert_weights.sum(dim=-1), torch.ones(2, 8), atol=1e-5))

    def test_top_k_indices_range(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=32, top_k=2, model_dim=16)
        bal_cfg = BalancerConfig()
        gate = SmartMoEGate(moe_cfg, bal_cfg)

        for _ in range(5):
            x = torch.randn(4, 16, 16)
            out = gate(x, training=False)
            idx = out.expert_indices
            self.assertTrue((idx >= 0).all())
            self.assertTrue((idx < 32).all())

    def test_force_assign_triggers(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=32, top_k=2, model_dim=32)
        bal_cfg = BalancerConfig(
            force_assign_ratio=0.5,
            utilization_threshold_low=0.3,
            utilization_warmup_steps=0,
        )
        gate = SmartMoEGate(moe_cfg, bal_cfg)
        tracker = MovingAverageUtilizationTracker(moe_cfg, bal_cfg)

        for _ in range(30):
            weights = torch.zeros(128, 32)
            weights[:, 0] = 1.0
            tracker.update(weights)

        x = torch.randn(4, 32, 32)
        out = gate(x, tracker=tracker, clusterer=None, lagrangian=None, training=True)
        self.assertGreater(out.routing_stats["force_assign_ratio"], 0)

    def test_gradient_through_gate(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=8, top_k=2, model_dim=16)
        bal_cfg = BalancerConfig()
        gate = SmartMoEGate(moe_cfg, bal_cfg)

        x = torch.randn(2, 4, 16, requires_grad=True)
        out = gate(x, training=True)
        loss = (out.expert_weights * out.expert_indices.float()).sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        for p in gate.parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad)


class TestDistributedMoELayer(unittest.TestCase):
    def test_end_to_end_forward(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=32, top_k=2, model_dim=32, expert_hidden_dim=128, experts_per_group=16)
        bal_cfg = BalancerConfig(force_assign_ratio=0.0)
        layer = DistributedMoELayer(moe_cfg, bal_cfg)

        x = torch.randn(2, 8, 32)
        out = layer(x, training=True)

        self.assertEqual(out.hidden_states.shape, (2, 8, 32))
        self.assertTrue(torch.isfinite(out.hidden_states).all())
        self.assertIsInstance(out.metrics, dict)

    def test_balance_recovery_from_extreme_imbalance(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=64, top_k=2, model_dim=32, expert_hidden_dim=64, experts_per_group=16)
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
            bias[~torch.isin(torch.arange(moe_cfg.num_experts), hot_idx)] = -2.0
            final_layer.bias.add_(bias)

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

            layer.step_balancers(out.gate_output)

            if "util/cv_sq" in out.metrics:
                cv_history.append(out.metrics["util/cv_sq"])
                if init_cv_sq is None and step > 10:
                    init_cv_sq = out.metrics["util/cv_sq"]
                if step >= 90:
                    final_cv_sq = out.metrics["util/cv_sq"]

        self.assertIsNotNone(init_cv_sq, "CV² metric should be present after step 10")
        self.assertIsNotNone(final_cv_sq, "CV² metric should be present at end")
        self.assertGreaterEqual(len(cv_history), 8,
                                f"Should have enough CV² samples, got {len(cv_history)}")

        print(f"\n[Balance Recovery Test] Initial CV²: {init_cv_sq:.4f}, Final CV²: {final_cv_sq:.4f}")
        print(f"[Balance Recovery Test] CV² history length: {len(cv_history)}")
        self.assertLess(final_cv_sq, init_cv_sq * 0.5,
                        f"Balance should improve significantly: {final_cv_sq:.4f} vs {init_cv_sq:.4f}")

    def test_full_backward_pass(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=16, top_k=2, model_dim=16, expert_hidden_dim=32, experts_per_group=8)
        bal_cfg = BalancerConfig(force_assign_ratio=0.2)
        layer = DistributedMoELayer(moe_cfg, bal_cfg)

        for _ in range(3):
            x = torch.randn(2, 4, 16, requires_grad=True)
            target = torch.randn_like(x)
            out = layer(x, training=True)
            loss = F.mse_loss(out.hidden_states, target) + out.total_aux_loss
            loss.backward()

            self.assertTrue(torch.isfinite(loss))

            has_grad_count = 0
            total_requires_grad = 0
            for name, p in layer.named_parameters():
                if p.requires_grad:
                    total_requires_grad += 1
                    if p.grad is not None:
                        has_grad_count += 1
                        self.assertTrue(torch.isfinite(p.grad).all(), f"Non-finite grad for {name}")

            self.assertGreater(has_grad_count, 0, "At least some params must have grad")
            self.assertGreaterEqual(total_requires_grad, 10)

            for p in layer.parameters():
                if p.grad is not None:
                    p.grad = None

    def test_get_balance_report(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=8, top_k=2, model_dim=16, expert_hidden_dim=32)
        bal_cfg = BalancerConfig()
        layer = DistributedMoELayer(moe_cfg, bal_cfg)

        x = torch.randn(2, 4, 16)
        for _ in range(15):
            layer(x, training=True)

        report = layer.get_balance_report()
        self.assertIn("utilization_cv", report)
        self.assertIn("balance_ratio", report)
        self.assertIn("underutilized_experts", report)
        self.assertIn("lambda_stats", report)


class TestIntegrationDeadlockResolution(unittest.TestCase):
    def test_force_assign_does_not_crash_training(self):
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

        with torch.no_grad():
            final = layer.gate.gate_proj[-1]
            bias = torch.zeros(64)
            bias[:2] = 10.0
            bias[2:] = -3.0
            final.bias.add_(bias)

        optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

        losses = []
        for step in range(50):
            optimizer.zero_grad(set_to_none=True)
            x = torch.randn(4, 16, 32)
            target = torch.randn_like(x)

            out = layer(x, training=True)
            task = F.mse_loss(out.hidden_states, target)
            total = task + out.total_aux_loss
            total.backward()

            torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
            optimizer.step()

            losses.append(task.item())
            self.assertTrue(torch.isfinite(total), f"Non-finite loss at step {step}")

        self.assertGreater(len(losses), 40)
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))

    def test_lagrangian_lambda_dynamics(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=32, top_k=2, model_dim=16, expert_hidden_dim=32)
        bal_cfg = BalancerConfig(
            utilization_threshold_low=0.2,
            utilization_threshold_high=2.0,
            lagrangian_lr=0.05,
            lagrangian_init_lambda=1.0,
            utilization_warmup_steps=0,
        )
        layer = DistributedMoELayer(moe_cfg, bal_cfg)

        with torch.no_grad():
            layer.gate.gate_proj[-1].bias[:1].add_(5.0)
            layer.gate.gate_proj[-1].bias[1:].sub_(2.0)

        lp_initial = layer.lagrangian_balancer._lambda_plus.clone()
        lp_under_initial = lp_initial[2:].mean()
        lm_over_initial = lp_initial[0]

        for _ in range(40):
            x = torch.randn(2, 8, 16)
            out = layer(x, training=True)
            layer.step_balancers(out.gate_output)

        lp_final = layer.lagrangian_balancer._lambda_plus
        lm_final = layer.lagrangian_balancer._lambda_minus

        self.assertGreater(lp_final[2:].mean().item(), lp_under_initial.item(),
                           "Lambda+ should increase for underutilized experts")
        self.assertGreater(lm_final[0].item(), lm_over_initial.item(),
                           "Lambda- should increase for overutilized expert")

    def test_cluster_based_redirect(self):
        set_seed()
        moe_cfg = MoEConfig(num_experts=32, top_k=2, model_dim=16, expert_hidden_dim=32)
        bal_cfg = BalancerConfig(
            cluster_update_interval=5,
            utilization_threshold_high=1.5,
            num_clusters=4,
            utilization_warmup_steps=0,
        )
        layer = DistributedMoELayer(moe_cfg, bal_cfg)

        with torch.no_grad():
            reps = layer.experts.get_expert_representations()
            layer.clusterer.compute_similarity_from_parameters(reps)
            layer.clusterer.cluster_experts_kmeans()
            layer.clusterer.mark_updated(step=0)

            for step in range(20):
                weights = torch.zeros(128, 32)
                weights[:, 0] = 1.0
                layer.utilization_tracker.update(weights)

        x = torch.randn(4, 8, 16)
        out = layer(x, training=True)
        redirect_ratio = out.gate_output.routing_stats.get("cluster_redirect_ratio", 0)
        print(f"[Cluster Redirect Test] Redirect ratio: {redirect_ratio:.4f}")
        self.assertIsInstance(redirect_ratio, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
