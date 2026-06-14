"""
Test that ablation CONCLUSION correctly switches to missing-data mode
when any metric field is absent on either side.

Simulates 3 scenarios:
  A) Both sides complete  -> should show HIGHLY EFFECTIVE / normal conclusion
  B) Baseline missing key fields  -> should show FAIL incomplete comparison, NO normal conclusion
  C) Ours missing key fields      -> same as B
  D) Both sides missing different fields  -> same as B

Validates:
  1. When missing data exists, "HIGHLY EFFECTIVE" / "EFFECTIVE" strings never appear
  2. Missing field names are printed explicitly
  3. ablation_summary.json correctly records missing_metrics
  4. Terminal output explicitly states "INCOMPLETE" and "ACTION REQUIRED"
"""

import io
import json
import math
import os
import sys
import tempfile
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple
from unittest.mock import patch

import torch


REQUIRED_KEYS = [
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


def make_complete_history(num_experts=32, with_balancer=True) -> Dict[str, List[float]]:
    """Build a realistic history dict with all 14 required keys populated."""
    h = {}
    if with_balancer:
        h["loss/task"] = [6.2, 6.1, 6.0]
        h["loss/aux"] = [80.0, 81.0, 80.5]
        h["util/util_cv_sq"] = [20.0, 5.0, 1.2]
        h["util/util_variance"] = [0.03, 0.01, 0.001]
        h["util/util_max"] = [0.95, 0.60, 0.22]
        h["util/util_min"] = [1e-6, 0.005, 0.015]
        h["util/balance_ratio"] = [1e6, 100.0, 14.6]
        h["util/underutilized_count"] = [0, 0, 0]
        h["util/long_tail_experts_count"] = [31, 5, 0]
        h["routing/unique_experts_used"] = [31, 31, 31]
        h["routing/top1_expert_share"] = [0.5, 0.52, 0.52]
        h["routing/force_ratio"] = [0.0, 0.05, 0.06]
        h["routing/redirect_ratio"] = [0.0, 0.30, 0.45]
        h["routing/both_ratio"] = [0.0, 0.0, 0.0]
    else:
        h["loss/task"] = [6.2, 6.2, 6.24]
        h["loss/aux"] = [0.0, 0.0, 0.0]
        h["util/util_cv_sq"] = [30.0, 31.0, 31.7]
        h["util/util_variance"] = [0.03, 0.031, 0.031]
        h["util/util_max"] = [0.98, 0.99, 0.995]
        h["util/util_min"] = [1e-6, 1e-6, 4e-6]
        h["util/balance_ratio"] = [2e5, 2e5, 252403.0]
        h["util/underutilized_count"] = [0, 0, 0]
        h["util/long_tail_experts_count"] = [31, 31, 31]
        h["routing/unique_experts_used"] = [31, 31, 31]
        h["routing/top1_expert_share"] = [0.5, 0.5, 0.5]
        h["routing/force_ratio"] = [0.0, 0.0, 0.0]
        h["routing/redirect_ratio"] = [0.0, 0.0, 0.0]
        h["routing/both_ratio"] = [0.0, 0.0, 0.0]
    return h


class StringLogger:
    """Minimal logger replacement that captures everything into a string buffer."""

    def __init__(self):
        self.buf = io.StringIO()
        self.records = {"info": [], "warning": [], "error": []}

    def info(self, msg):
        self.buf.write(msg + "\n")
        self.records["info"].append(msg)

    def warning(self, msg):
        self.buf.write(msg + "\n")
        self.records["warning"].append(msg)

    def error(self, msg):
        self.buf.write(msg + "\n")
        self.records["error"].append(msg)

    def getvalue(self) -> str:
        return self.buf.getvalue()


def test_scenarios():
    import importlib
    import train_1024expert as tmod

    print("\n" + "=" * 80)
    print("TESTING print_ablation_summary CONCLUSION logic with mocked histories")
    print("=" * 80)

    scenarios = []

    # --- Scenario A: Both complete ---
    baseline_ok = make_complete_history(with_balancer=False)
    ours_ok = make_complete_history(with_balancer=True)
    scenarios.append(("A: Both sides COMPLETE", baseline_ok, ours_ok, "complete"))

    # --- Scenario B: Baseline missing CV^2 and long_tail ---
    baseline_miss = make_complete_history(with_balancer=False)
    del baseline_miss["util/util_cv_sq"]
    del baseline_miss["util/long_tail_experts_count"]
    scenarios.append(("B: Baseline missing CV^2 & long_tail", baseline_miss, ours_ok, "missing"))

    # --- Scenario C: Ours missing task_loss and force_ratio ---
    ours_miss = make_complete_history(with_balancer=True)
    del ours_miss["loss/task"]
    ours_miss["routing/force_ratio"] = []  # empty list = effectively missing
    scenarios.append(("C: Ours missing task_loss & force_ratio", baseline_ok, ours_miss, "missing"))

    # --- Scenario D: Both missing different fields ---
    baseline_miss2 = make_complete_history(with_balancer=False)
    ours_miss2 = make_complete_history(with_balancer=True)
    del baseline_miss2["util/balance_ratio"]
    del ours_miss2["util/util_min"]
    del ours_miss2["routing/redirect_ratio"]
    scenarios.append(("D: Both missing different fields", baseline_miss2, ours_miss2, "missing"))

    # --- Scenario E: Ours missing ALL key metrics ---
    ours_miss_all = make_complete_history(with_balancer=True)
    for k in list(ours_miss_all.keys()):
        if k != "loss/aux":
            ours_miss_all[k] = []
    scenarios.append(("E: Ours missing nearly all fields", baseline_ok, ours_miss_all, "missing"))

    all_passed = True
    summary_results = []

    for name, baseline, ours, expected_mode in scenarios:
        print(f"\n--- Running {name} ---")
        tmpdir = tempfile.mkdtemp(prefix="ablation_test_")
        logger = StringLogger()

        with patch.object(tmod, "logger", logger):
            try:
                info = tmod.print_ablation_summary(baseline, ours, tmpdir)
            except Exception as e:
                print(f"  EXCEPTION during print_ablation_summary: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False
                continue

        output = logger.getvalue()
        records = logger.records

        has_normal_conclusion = any(
            any(s in r for s in ["HIGHLY EFFECTIVE", "Load balancer EFFECTIVE", "SOME effect"])
            for r in records["info"]
        )
        has_incomplete = any(
            "INCOMPLETE" in r or "incomplete" in r.lower()
            for r in records["error"] + records["warning"]
        )
        has_fail_tag = any(
            "[FAIL]" in r for r in records["error"] + records["warning"]
        )
        has_action = any(
            "ACTION REQUIRED" in r for r in records["error"]
        )

        summary_path = os.path.join(tmpdir, "ablation_summary.json")
        summary_exists = os.path.exists(summary_path)
        summary_has_missing = False
        summary_missing_list = []
        if summary_exists:
            with open(summary_path, "r") as f:
                summary = json.load(f)
            summary_has_missing = len(summary.get("missing_metrics", [])) > 0
            summary_missing_list = summary.get("missing_metrics", [])

        # Also check returned dict
        ret_has_missing = info.get("has_missing", False)
        ret_missing_count = len(info.get("missing_metrics", []))

        passed = True

        if expected_mode == "complete":
            if not has_normal_conclusion:
                print(f"  [FAIL] Expected normal conclusion, but not found in output")
                passed = False
            if has_incomplete or has_fail_tag:
                print(f"  [FAIL] Expected clean run, but found INCOMPLETE/FAIL tag")
                passed = False
            if summary_has_missing:
                print(f"  [FAIL] summary.json shows missing_metrics but shouldn't")
                passed = False
            if ret_has_missing:
                print(f"  [FAIL] returned has_missing=True but shouldn't")
                passed = False
        else:
            if has_normal_conclusion:
                print(f"  [FAIL] Found normal 'HIGHLY EFFECTIVE / EFFECTIVE / SOME' conclusion in MISSING scenario!")
                passed = False
            if not has_incomplete:
                print(f"  [FAIL] Missing scenario but 'INCOMPLETE' not in error/warning output")
                passed = False
            if not has_fail_tag:
                print(f"  [FAIL] Missing scenario but no [FAIL] tag found")
                passed = False
            if not has_action:
                print(f"  [FAIL] Missing scenario but 'ACTION REQUIRED' not printed")
                passed = False
            if not summary_has_missing:
                print(f"  [FAIL] summary.json missing_metrics should be non-empty")
                passed = False
            if not ret_has_missing:
                print(f"  [FAIL] returned has_missing=False but should be True")
                passed = False
            if ret_missing_count < 1:
                print(f"  [FAIL] returned missing_metrics is empty")
                passed = False
            # Specific checks: print which fields were actually reported missing
            print(f"  Missing count (returned dict): {ret_missing_count}")
            print(f"  Missing list (from JSON):")
            for m in summary_missing_list:
                print(f"    * {m}")
            print(f"  Logger ERROR lines with missing info:")
            for r in records["error"]:
                if "Missing on" in r or "SKIPPED" in r or "INCOMPLETE" in r or "ACTION" in r:
                    print(f"    > {r.strip()}")

        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} Scenario {name}")
        summary_results.append((name, passed))
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    for name, passed in summary_results:
        print(f"  {'[OK]' if passed else '[FAIL]'}  {name}")

    if all_passed:
        print("\nALL SCENARIOS PASSED.")
        return 0
    else:
        print("\nSOME SCENARIOS FAILED — see above.")
        return 1


if __name__ == "__main__":
    sys.exit(test_scenarios())
