#!/usr/bin/env python3
"""Reproducible predictive-residency benchmark with rotated arms and medians.

This harness does not drop caches. Runs are therefore not labeled cold unless
the operator separately controls and records cache state. Simulation is not
included; use `ramforge simulate` for deterministic simulation.
"""
import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


ARMS = {
    "baseline": {"prediction": False, "prefetch": False},
    "reactive": {"prediction": False, "prefetch": True},
    "predictive": {
        "prediction": True,
        "min_conf": 40,
        "window": "64M",
        "speculative": "16M",
    },
    "predictive_conservative": {
        "prediction": True,
        "min_conf": 70,
        "window": "32M",
        "speculative": "8M",
    },
    "predictive_aggressive": {
        "prediction": True,
        "min_conf": 25,
        "window": "128M",
        "speculative": "64M",
    },
}


def git_head(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def command_for(binary, model, profile, history, target, ctx, prompt, tokens, arm, enforce, timeout):
    cfg = ARMS[arm]
    json_path = None  # filled by caller
    cmd = [
        binary,
        "run",
        model,
        "--ram",
        target,
        "--ctx",
        str(ctx),
        "--no-repack",
        "--no-prefetch",
        "--residency-policy",
        "observe",
        "--residency-target",
        target,
        "--graph-trace",
        "--graph-trace-events",
        "16384",
        "--prompt",
        prompt,
        "--max-tokens",
        str(tokens),
        "--temperature",
        "0",
        "--backend",
        "forgecore",
    ]
    if cfg["prediction"]:
        cmd += [
            "--prediction",
            "--prediction-min-confidence",
            str(cfg["min_conf"]),
            "--prediction-window",
            cfg["window"],
            "--prediction-speculative",
            cfg["speculative"],
            "--residency-profile-load",
            profile,
        ]
        if history:
            cmd += ["--prediction-history-load", history]
    elif cfg["prefetch"]:
        cmd += ["--residency-prefetch", "64M"]
    if enforce:
        cmd += ["--enforce-ram", "--enforce-timeout", str(timeout)]
    return cmd


def parse_report(path, arm, workload, repeat, order, wall_ns, status, stderr):
    if status == 0 and Path(path).is_file():
        report = json.loads(Path(path).read_text())
        prediction = report.get("prediction") or {}
        metrics = prediction.get("metrics") or {}
        plan = prediction.get("plan") or {}
        staging = prediction.get("staging") or {}
        history = prediction.get("history") or {}
        before = (report.get("measurements") or {}).get("before") or {}
        after = (report.get("measurements") or {}).get("after") or {}
        timing = report.get("timing") or {}
        faults = report.get("process_faults") or {}
        app = report.get("application") or {}
        native = report.get("native_lifetime") or {}
        return {
            "arm": arm,
            "workload": workload,
            "repeat": repeat,
            "order": order,
            "status": status,
            "wall_ns": wall_ns,
            "ttft_ns": timing.get("ttft_ns"),
            "decode_ns": timing.get("decode_ns"),
            "total_ns": timing.get("total_ns"),
            "output_fingerprint": timing.get("output_fingerprint"),
            "rss_before": before.get("process_rss_bytes"),
            "rss_after": after.get("process_rss_bytes"),
            "mapping_before": before.get("mapping_rss_bytes"),
            "mapping_after": after.get("mapping_rss_bytes"),
            "mapping_delta": report.get("estimated_vs_observed", {}).get("observed_mapping_delta_bytes")
            if isinstance(report.get("estimated_vs_observed"), dict)
            else None,
            "minor_faults_before": before.get("process_minor_faults"),
            "minor_faults_after": after.get("process_minor_faults"),
            "major_faults_before": before.get("process_major_faults"),
            "major_faults_after": after.get("process_major_faults"),
            "minor_faults_before_load": faults.get("before_load_minor"),
            "major_faults_before_load": faults.get("before_load_major"),
            "cgroup_before": before.get("cgroup_current_bytes"),
            "cgroup_after": after.get("cgroup_current_bytes"),
            "cgroup_peak": after.get("cgroup_peak_bytes"),
            "cgroup_events": after.get("cgroup_events"),
            "rss_peak": after.get("process_rss_peak_bytes"),
            "minor_faults_delta": (after.get("process_minor_faults") or 0)
            - (faults.get("before_load_minor") or 0)
            if faults.get("before_load_minor") is not None
            and after.get("process_minor_faults") is not None
            else None,
            "major_faults_delta": (after.get("process_major_faults") or 0)
            - (faults.get("before_load_major") or 0)
            if faults.get("before_load_major") is not None
            and after.get("process_major_faults") is not None
            else None,
            "prediction_count": prediction.get("prediction_count"),
            "prediction_matched": metrics.get("matched_observed"),
            "prediction_within_horizon": metrics.get("matched_within_horizon"),
            "prediction_precision_bps": metrics.get("precision_bps"),
            "prediction_recall_bps": metrics.get("recall_bps"),
            "prediction_fpr_bps": metrics.get("false_positive_rate_bps"),
            "bucket_high_precision_bps": (metrics.get("bucket_high") or {}).get("precision_bps"),
            "bucket_medium_precision_bps": (metrics.get("bucket_medium") or {}).get("precision_bps"),
            "bucket_low_precision_bps": (metrics.get("bucket_low") or {}).get("precision_bps"),
            "prefetch_requested_resources": metrics.get("prefetch_requested_resources"),
            "prefetch_requested_bytes": metrics.get("prefetch_requested_bytes"),
            "prefetch_accepted_bytes": metrics.get("prefetch_accepted_bytes"),
            "staging_latency_ns": staging.get("elapsed_ns"),
            "prefetch_accepted_resources": metrics.get("prefetch_accepted_resources"),
            "prefetch_followed": metrics.get("prefetch_followed_by_observed_access"),
            "prefetch_not_followed": metrics.get("prefetch_not_followed_by_observed_access"),
            "prefetch_unknown": metrics.get("prefetch_benefit_unknown_resources"),
            "duplicate_or_overlap_avoided": metrics.get("duplicate_or_overlap_avoided"),
            "deferred": metrics.get("deferred"),
            "rejected": metrics.get("rejected"),
            "history_hits": history.get("hits"),
            "history_misses": history.get("misses"),
            "history_wasted_bytes": history.get("wasted_bytes"),
            "native_contract": native.get("contract_version"),
            "native_quiescent": native.get("quiescent"),
            "released_bytes": app.get("native_released_bytes_submitted"),
            "stderr": stderr,
            "json": str(path),
        }
    return {
        "arm": arm,
        "workload": workload,
        "repeat": repeat,
        "order": order,
        "status": status,
        "wall_ns": wall_ns,
        "stderr": stderr,
        "json": str(path),
    }


def median(rows, key):
    values = [r[key] for r in rows if r.get(key) not in (None, "")]
    return statistics.median(values) if values else None


def spread(rows, key):
    values = [r[key] for r in rows if r.get(key) not in (None, "")]
    if not values:
        return None
    return {"min": min(values), "max": max(values), "median": statistics.median(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("FORGE_TEST_MODEL", ""))
    parser.add_argument("--profile", default=os.environ.get("PROFILE", ""))
    parser.add_argument("--history", default=os.environ.get("PREDICTION_HISTORY", ""))
    parser.add_argument("--binary", default=os.environ.get("RAMFORGE_BIN", "target/debug/ramforge"))
    parser.add_argument("--out", default=os.environ.get("RAMFORGE_BENCH_OUT", "prediction-bench-out"))
    parser.add_argument("--target", default="192M")
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    if not args.model or not args.profile:
        parser.error("--model and --profile are required")
    if not Path(args.binary).is_file():
        parser.error(f"binary not found: {args.binary}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    workloads = [
        ("short-short", "The capital of France is", 1, 256),
        ("short-long", "The capital of France is", 8, 512),
        ("long-short", "The quick brown fox jumps over the lazy dog. " * 20, 1, 512),
        ("alternate", "Write a short Rust function that adds two numbers.", 4, 512),
    ]
    rows = []
    for workload_index, (name, prompt, tokens, ctx) in enumerate(workloads):
        for repeat in range(1, args.repeats + 1):
            arm_names = list(ARMS)
            shift = (repeat + workload_index) % len(arm_names)
            arm_names = arm_names[shift:] + arm_names[:shift]
            for order, arm in enumerate(arm_names, start=1):
                stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                stem = f"{stamp}-{name}-{arm}-r{repeat}"
                json_path = out / f"{stem}.json"
                err_path = out / f"{stem}.stderr"
                cmd = command_for(
                    args.binary,
                    args.model,
                    args.profile,
                    args.history,
                    args.target,
                    ctx,
                    prompt,
                    tokens,
                    arm,
                    args.enforce,
                    args.timeout,
                )
                cmd += ["--residency-report-json", str(json_path)]
                start = time.time_ns()
                with err_path.open("w") as err:
                    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=err)
                wall = time.time_ns() - start
                rows.append(
                    parse_report(
                        json_path, arm, name, repeat, order, wall, proc.returncode, str(err_path)
                    )
                )

    if rows:
        fields = sorted({key for row in rows for key in row})
        with (out / "results.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "benchmark": "ramforge.prediction_bench.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": args.model,
        "profile": args.profile,
        "history": args.history or None,
        "target": args.target,
        "enforce": args.enforce,
        "repeats": args.repeats,
        "cache_controlled": False,
        "cache_note": "No cache-drop was performed; results are not labeled cold.",
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "ramforge": git_head(Path(__file__).resolve().parents[1]),
        "forgecore": git_head(Path(__file__).resolve().parents[2] / "forgeCORE"),
        "native_pin": "7fe450e19305b828c199d602c23a8337aaa1f03b",
        "workloads": [w[0] for w in workloads],
        "arms": list(ARMS),
        "runs": len(rows),
        "failures": sum(1 for row in rows if row.get("status") != 0),
        "summary": {},
    }
    for workload in summary["workloads"]:
        summary["summary"][workload] = {}
        for arm in ARMS:
            selected = [
                row for row in rows if row["workload"] == workload and row["arm"] == arm
            ]
            summary["summary"][workload][arm] = {
                "runs": len(selected),
                "failures": sum(1 for row in selected if row.get("status") != 0),
                "ttft_ns": spread(selected, "ttft_ns"),
                "total_ns": spread(selected, "total_ns"),
                "rss_after": spread(selected, "rss_after"),
                "rss_peak": spread(selected, "rss_peak"),
                "cgroup_peak": spread(selected, "cgroup_peak"),
                "minor_faults_after": spread(selected, "minor_faults_after"),
                "major_faults_after": spread(selected, "major_faults_after"),
                "minor_faults_delta": spread(selected, "minor_faults_delta"),
                "major_faults_delta": spread(selected, "major_faults_delta"),
                "mapping_delta": spread(selected, "mapping_delta"),
                "prediction_count": median(selected, "prediction_count"),
                "prediction_matched": median(selected, "prediction_matched"),
                "prediction_within_horizon": median(selected, "prediction_within_horizon"),
                "prediction_precision_bps": median(selected, "prediction_precision_bps"),
                "prediction_recall_bps": median(selected, "prediction_recall_bps"),
                "prediction_fpr_bps": median(selected, "prediction_fpr_bps"),
                "bucket_high_precision_bps": median(selected, "bucket_high_precision_bps"),
                "bucket_medium_precision_bps": median(selected, "bucket_medium_precision_bps"),
                "bucket_low_precision_bps": median(selected, "bucket_low_precision_bps"),
                "prefetch_requested_bytes": median(selected, "prefetch_requested_bytes"),
                "prefetch_accepted_bytes": median(selected, "prefetch_accepted_bytes"),
                "staging_latency_ns": spread(selected, "staging_latency_ns"),
                "prefetch_followed": median(selected, "prefetch_followed"),
                "prefetch_not_followed": median(selected, "prefetch_not_followed"),
                "prefetch_unknown": median(selected, "prefetch_unknown"),
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    lines = [
        "RAMforge prediction benchmark (simulation not included)",
        f"model={args.model}",
        f"profile={args.profile} history={args.history or 'none'}",
        f"target={args.target} enforce={args.enforce} repeats={args.repeats}",
        f"cache_controlled={summary['cache_controlled']} ({summary['cache_note']})",
        f"runs={summary['runs']} failures={summary['failures']}",
    ]
    for workload in summary["workloads"]:
        lines.append(f"\n[{workload}]")
        for arm in ARMS:
            entry = summary["summary"][workload][arm]
            ttft = entry.get("ttft_ns") or {}
            major = entry.get("major_faults_delta") or {}
            lines.append(
                "  {arm}: ttft_median_ns={ttft} ttft_min={tmin} ttft_max={tmax} "
                "major_fault_delta_median={major} accepted_bytes={accepted} "
                "matched={matched} within_horizon={within} precision_bps={precision} "
                "recall_bps={recall} staging_latency_ns={staging}".format(
                    arm=arm,
                    ttft=ttft.get("median"),
                    tmin=ttft.get("min"),
                    tmax=ttft.get("max"),
                    major=major.get("median"),
                    accepted=entry.get("prefetch_accepted_bytes"),
                    matched=entry.get("prediction_matched"),
                    within=entry.get("prediction_within_horizon"),
                    precision=entry.get("prediction_precision_bps"),
                    recall=entry.get("prediction_recall_bps"),
                    staging=(entry.get("staging_latency_ns") or {}).get("median"),
                )
            )
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if summary["failures"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
