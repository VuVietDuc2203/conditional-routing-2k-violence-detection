
#!/usr/bin/env python3
"""Check that v10 headline numbers are traceable to sealed evidence.

This is deliberately narrower than a prose linter: it verifies the locked
offline endpoints and the workload-level matched-replay values that are most
likely to be copied into the abstract, Results, Discussion, or Conclusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


WORKLOADS = ("normal", "mixed", "kinetic")
MODES = ("m3_gated", "m1_dense_s1")


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--vi", type=Path, required=True)
    parser.add_argument("--sourcewise", type=Path, required=True)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--matched-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tex = args.tex.read_text(encoding="utf-8")
    vi = args.vi.read_text(encoding="utf-8")
    sourcewise = json.loads(args.sourcewise.read_text(encoding="utf-8"))
    matched = json.loads(args.matched.read_text(encoding="utf-8"))
    table = args.matched_table.read_text(encoding="utf-8")

    failures: list[str] = []
    checks: dict[str, object] = {}
    expected_offline = {
        "m1_dense": (509, 526, "96.77", "96,77"),
        "m3_crowd": (502, 526, "95.44", "95,44"),
        "routed_offline": (495, 526, "94.11", "94,11"),
    }
    observed = sourcewise["overall_locked_endpoints"]
    for endpoint, (correct, n, en_pct, vi_pct) in expected_offline.items():
        row = observed[endpoint]
        if int(row["correct"]) != correct or int(row["n"]) != n:
            failures.append(f"locked endpoint changed: {endpoint}")
        if en_pct not in tex:
            failures.append(f"English manuscript lacks locked percentage {en_pct}: {endpoint}")
        if vi_pct not in vi:
            failures.append(f"Vietnamese manuscript lacks locked percentage {vi_pct}: {endpoint}")
    checks["offline_locked_counts"] = {
        key: {"correct": int(observed[key]["correct"]), "n": int(observed[key]["n"])}
        for key in expected_offline
    }

    rows = {(str(row["mode"]), str(row["workload"])): row for row in matched["rows"]}
    expected_cells = {(mode, workload) for mode in MODES for workload in WORKLOADS}
    if set(rows) != expected_cells:
        failures.append("matched replay does not contain the six required mode/workload rows")
    replay_checks: dict[str, object] = {}
    for mode, workload in sorted(expected_cells):
        if (mode, workload) not in rows:
            continue
        row = rows[(mode, workload)]
        if int(row["n"]) != 3:
            failures.append(f"matched replay n is not three: {mode}/{workload}")
        expected_tokens = {
            "throughput_mean": f"{float(row['achieved_analysis_fps']):.3f}",
            "throughput_sd": f"{float(row['achieved_analysis_fps_sample_sd']):.3f}",
            "q_update": pct(float(row["q_update"])),
            "deadline_miss": pct(float(row["deadline_miss_rate"])),
            "gpu_energy": f"{float(row['gpu_board_energy_per_analyzed_update_j']):.2f}",
        }
        missing = [name for name, token in expected_tokens.items() if token not in table]
        if missing:
            failures.append(f"matched table lacks {mode}/{workload}: {', '.join(missing)}")
        replay_checks[f"{mode}/{workload}"] = expected_tokens

    narrative = matched.get("narrative_summary", {})
    narrative_tokens = narrative.get("tokens", {}) if isinstance(narrative, dict) else {}
    required_token_names = (
        "routed_throughput",
        "continuous_throughput",
        "routed_calls_per_100",
        "continuous_calls_per_100",
        "routed_deadline_miss_percent",
        "continuous_deadline_miss_percent",
        "routed_board_energy_j_per_update",
        "continuous_board_energy_j_per_update",
        "throughput_ratio",
        "classifier_call_reduction_percent",
        "board_energy_reduction_percent",
    )
    narrative_checks: dict[str, object] = {}
    for language, content in (("en", tex), ("vi", vi)):
        language_tokens = narrative_tokens.get(language, {}) if isinstance(narrative_tokens, dict) else {}
        missing_names = [name for name in required_token_names if not language_tokens.get(name)]
        if missing_names:
            failures.append(f"verified narrative tokens missing for {language}: {', '.join(missing_names)}")
            continue
        absent = [name for name in required_token_names if str(language_tokens[name]) not in content]
        if absent:
            failures.append(f"{language} manuscript lacks verified matched-replay narrative tokens: {', '.join(absent)}")
        narrative_checks[language] = {name: language_tokens[name] for name in required_token_names}

    stale_tokens = ("14.660", "14.186", "12.493", "14,66", "14,19", "12,49")
    stale_present = [token for token in stale_tokens if token in tex or token in vi]
    if stale_present:
        failures.append(f"stale routed-only replay values remain in manuscript: {stale_present}")

    prohibited = "stage4_prime_paired_accuracy_analysis.json"
    if prohibited in tex or prohibited in vi:
        failures.append("manuscript cites excluded pre-amendment accuracy artifact")
    if "source-wise results" in tex and "Supplementary Tables~S1--S2" not in tex:
        failures.append("English source-wise reporting lacks its supplementary pointer")

    checks["matched_table_tokens"] = replay_checks
    checks["matched_narrative_tokens"] = narrative_checks
    checks["pre_amendment_artifact_absent"] = prohibited not in tex and prohibited not in vi
    payload = {"status": "pass" if not failures else "fail", "failures": failures, "checks": checks}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("numeric consistency audit failed; see output JSON")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

