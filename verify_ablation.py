"""
Minimal acceptance test for MoE load balancer ablation study.

Runs a small-scale ablation experiment and validates:
  1. Both Baseline (No Bal) and Ours (With Bal) produce all expected metrics
  2. All metric values are valid numbers (no NaN, no missing)
  3. Baseline routing proportions (force/redirect/both) are 0%
  4. Ours shows meaningful load balancing improvement
  5. Summary JSON file is properly structured

Usage:
    python verify_ablation.py [--quick] [--num_experts N] [--max_steps S]
"""

import argparse
import json
import math
import os
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from train_1024expert import run_ablation


REQUIRED_METRICS_KEYS = [
    "loss/task",
    "loss/aux",
    "util/util_cv_sq",
    "util/util_variance",
    "util/util_max",
    "util/util_min",
    "util/balance_ratio",
    "util/underutilized_count",
    "util/long_tail_experts_count",
    "routing/unique_experts_used",
    "routing/top1_expert_share",
    "routing/force_ratio",
    "routing/redirect_ratio",
    "routing/both_ratio",
]

ROUTING_RATIO_KEYS = [
    "routing/force_ratio",
    "routing/redirect_ratio",
    "routing/both_ratio",
]


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    details: Dict = field(default_factory=dict)


def is_valid_number(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    if isinstance(v, float) and math.isinf(v):
        return False
    return isinstance(v, (int, float))


def check_history_has_key(history: Dict, key: str, label: str) -> CheckResult:
    if key not in history:
        return CheckResult(
            name=f"{label}: has {key}",
            passed=False,
            message=f"Key '{key}' missing from {label} history",
        )
    vals = history[key]
    if not isinstance(vals, list) or len(vals) == 0:
        return CheckResult(
            name=f"{label}: {key} has data",
            passed=False,
            message=f"Key '{key}' in {label} is empty or not a list",
            details={"type": type(vals).__name__, "len": len(vals) if isinstance(vals, list) else "N/A"},
        )
    last_val = vals[-1]
    if not is_valid_number(last_val):
        return CheckResult(
            name=f"{label}: {key} last value valid",
            passed=False,
            message=f"Last value of '{key}' in {label} is not a valid number: {last_val}",
        )
    return CheckResult(
        name=f"{label}: {key}",
        passed=True,
        message=f"OK - last = {last_val:.6f}",
        details={"last_value": last_val},
    )


def check_baseline_routing_ratios_are_zero(baseline: Dict) -> List[CheckResult]:
    results = []
    for key in ROUTING_RATIO_KEYS:
        if key not in baseline or not baseline[key]:
            results.append(CheckResult(
                name=f"Baseline {key} == 0",
                passed=False,
                message=f"Key '{key}' missing from baseline",
            ))
            continue
        vals = baseline[key]
        non_zero = [v for v in vals if is_valid_number(v) and abs(v) > 1e-6]
        if non_zero:
            results.append(CheckResult(
                name=f"Baseline {key} == 0",
                passed=False,
                message=f"Baseline {key} has {len(non_zero)} non-zero values, "
                        f"max = {max(non_zero):.6f} (should be 0 with balancer OFF)",
                details={"max_non_zero": max(non_zero)},
            ))
        else:
            results.append(CheckResult(
                name=f"Baseline {key} == 0",
                passed=True,
                message=f"OK - all {len(vals)} values are 0",
            ))
    return results


def check_summary_json(output_dir: str) -> List[CheckResult]:
    results = []
    summary_path = os.path.join(output_dir, "ablation_summary.json")
    if not os.path.exists(summary_path):
        results.append(CheckResult(
            name="Summary JSON exists",
            passed=False,
            message=f"File not found: {summary_path}",
        ))
        return results

    try:
        with open(summary_path, "r") as f:
            summary = json.load(f)
    except Exception as e:
        results.append(CheckResult(
            name="Summary JSON parseable",
            passed=False,
            message=f"Failed to parse: {e}",
        ))
        return results

    if "metrics" not in summary:
        results.append(CheckResult(
            name="Summary has 'metrics' field",
            passed=False,
            message="'metrics' key missing from summary",
        ))
        return results

    metrics = summary["metrics"]
    for key in REQUIRED_METRICS_KEYS:
        if key not in metrics:
            results.append(CheckResult(
                name=f"Summary metrics has {key}",
                passed=False,
                message=f"Key '{key}' missing from summary metrics",
            ))
            continue
        m = metrics[key]
        for field in ["name", "baseline", "ours", "mode"]:
            if field not in m:
                results.append(CheckResult(
                    name=f"Summary metrics[{key}].{field}",
                    passed=False,
                    message=f"Field '{field}' missing for key '{key}'",
                ))
                continue
            if field in ["baseline", "ours"] and not is_valid_number(m[field]):
                results.append(CheckResult(
                    name=f"Summary metrics[{key}].{field} valid",
                    passed=False,
                    message=f"{field} = {m[field]} is not a valid number",
                ))

    if not any(not r.passed for r in results):
        results.append(CheckResult(
            name="Summary JSON structure",
            passed=True,
            message=f"OK - all {len(REQUIRED_METRICS_KEYS)} metrics present and valid",
        ))

    return results


def check_load_balancing_works(baseline: Dict, ours: Dict) -> CheckResult:
    b_cv = baseline.get("util/util_cv_sq", [])
    o_cv = ours.get("util/util_cv_sq", [])
    if not b_cv or not o_cv:
        return CheckResult(
            name="Load balancing improves CV^2",
            passed=False,
            message="CV^2 data missing from one or both runs",
        )
    b_last = b_cv[-1]
    o_last = o_cv[-1]
    if not is_valid_number(b_last) or not is_valid_number(o_last):
        return CheckResult(
            name="Load balancing improves CV^2",
            passed=False,
            message=f"Invalid CV^2 values: baseline={b_last}, ours={o_last}",
        )
    if o_last < b_last * 0.5:
        improvement = (1 - o_last / max(b_last, 1e-9)) * 100
        return CheckResult(
            name="Load balancing improves CV^2",
            passed=True,
            message=f"OK - CV^2 reduced by {improvement:.1f}% ({b_last:.4f} -> {o_last:.4f})",
            details={"baseline_cv": b_last, "ours_cv": o_last, "improvement_pct": improvement},
        )
    else:
        return CheckResult(
            name="Load balancing improves CV^2",
            passed=False,
            message=f"CV^2 not improved enough: {b_last:.4f} -> {o_last:.4f} "
                    f"(need >50% reduction, got {(1-o_last/b_last)*100:.1f}%)",
        )


def main():
    parser = argparse.ArgumentParser(description="MoE Load Balancer Ablation Acceptance Test")
    parser.add_argument("--num_experts", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=60)
    parser.add_argument("--model_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--report_interval", type=int, default=20)
    parser.add_argument("--cluster_interval", type=int, default=30)
    parser.add_argument("--cpu", action="store_true", default=True)
    parser.add_argument("--quick", action="store_true", help="Use even smaller params for CI")
    args = parser.parse_args()

    if args.quick:
        args.num_experts = 16
        args.max_steps = 30
        args.model_dim = 32
        args.batch_size = 2
        args.seq_len = 8
        args.warmup_steps = 5
        args.report_interval = 10
        args.cluster_interval = 15

    output_dir = tempfile.mkdtemp(prefix="moe_ablation_test_")
    print(f"Output dir: {output_dir}")

    class Config:
        def __init__(self):
            self.num_experts = args.num_experts
            self.model_dim = args.model_dim
            self.num_layers = 1
            self.num_heads = 2
            self.vocab_size = 512
            self.batch_size = args.batch_size
            self.seq_len = args.seq_len
            self.max_steps = args.max_steps
            self.warmup_steps = args.warmup_steps
            self.learning_rate = 1e-4
            self.grad_clip = 0.5
            self.report_interval = args.report_interval
            self.cluster_interval = args.cluster_interval
            self.output_dir = output_dir
            self.cpu = args.cpu
            self.seed = 42
            self.inject_imbalance = True
            self.inject_ratio = 0.05

    config = Config()

    all_checks: List[CheckResult] = []

    print("\n" + "=" * 80)
    print("STEP 1: Run ablation experiment")
    print("=" * 80)

    try:
        baseline, ours = run_ablation(config)
        print("Ablation run completed successfully")
    except Exception as e:
        print(f"ERROR: Ablation run failed: {e}")
        traceback.print_exc()
        all_checks.append(CheckResult(
            name="Ablation run completes",
            passed=False,
            message=f"Exception: {e}",
        ))
        _print_results(all_checks)
        sys.exit(1)

    all_checks.append(CheckResult(
        name="Ablation run completes",
        passed=True,
        message="OK",
    ))

    print("\n" + "=" * 80)
    print("STEP 2: Check Baseline (No Balancer) metrics")
    print("=" * 80)

    for key in REQUIRED_METRICS_KEYS:
        all_checks.append(check_history_has_key(baseline, key, "Baseline"))

    print("\n" + "=" * 80)
    print("STEP 3: Check Ours (With Balancer) metrics")
    print("=" * 80)

    for key in REQUIRED_METRICS_KEYS:
        all_checks.append(check_history_has_key(ours, key, "Ours"))

    print("\n" + "=" * 80)
    print("STEP 4: Check Baseline routing ratios are exactly 0")
    print("=" * 80)

    all_checks.extend(check_baseline_routing_ratios_are_zero(baseline))

    print("\n" + "=" * 80)
    print("STEP 5: Check Summary JSON structure")
    print("=" * 80)

    all_checks.extend(check_summary_json(output_dir))

    print("\n" + "=" * 80)
    print("STEP 6: Check load balancing actually works")
    print("=" * 80)

    all_checks.append(check_load_balancing_works(baseline, ours))

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    passed = _print_results(all_checks)

    print(f"\nOutput directory: {output_dir}")
    print(f"  - ablation_baseline.json: {os.path.exists(os.path.join(output_dir, 'ablation_baseline.json'))}")
    print(f"  - ablation_ours.json: {os.path.exists(os.path.join(output_dir, 'ablation_ours.json'))}")
    print(f"  - ablation_summary.json: {os.path.exists(os.path.join(output_dir, 'ablation_summary.json'))}")

    sys.exit(0 if passed else 1)


def _print_results(all_checks: List[CheckResult]) -> bool:
    passed_count = sum(1 for c in all_checks if c.passed)
    failed_count = len(all_checks) - passed_count

    print(f"\nTotal checks: {len(all_checks)}")
    print(f"  Passed: {passed_count}")
    print(f"  Failed: {failed_count}")

    if failed_count > 0:
        print("\n[FAIL] FAILED CHECKS:")
        for c in all_checks:
            if not c.passed:
                print(f"  - [{c.name}] {c.message}")
                if c.details:
                    for k, v in c.details.items():
                        print(f"      {k}: {v}")

    if passed_count == len(all_checks):
        print("\n[OK] ALL CHECKS PASSED!")
        print("\nSummary of passing checks:")
        for c in all_checks:
            print(f"  [OK] [{c.name}] {c.message}")
    else:
        print(f"\n[WARN] {failed_count} check(s) failed.")

    return passed_count == len(all_checks)


if __name__ == "__main__":
    main()
