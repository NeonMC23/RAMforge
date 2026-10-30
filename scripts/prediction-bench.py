#!/usr/bin/env python3
"""Reproducible predictive-residency benchmark with rotated arms and medians.

This harness does not drop caches. Runs are therefore not labeled cold unless
the operator separately controls and records cache state. Simulation is not
included; use `ramforge simulate` for deterministic simulation.
"""
import argparse
import ctypes
import csv
import hashlib
import json
import mmap
import os
import platform
import shutil
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


def _probe_residency(path):
    """Verification only. mmap() twice and mincore() do not fault pages in.

    Returns page-cache residency counts for the isolated copy before RAMforge
    starts. This is not proof of physical reclamation or PSS.
    """
    result = {
        "pages": None,
        "resident_pages": None,
        "nonresident_pages": None,
        "unknown_pages": None,
        "error": None,
    }
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as error:
        result["error"] = f"open: {error}"
        return result
    address = None
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            result.update(pages=0, resident_pages=0, nonresident_pages=0, unknown_pages=0)
            return result
        page = mmap.PAGESIZE
        pages = (size + page - 1) // page
        length = pages * page
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        address = libc.mmap(
            None, length, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0
        )
        if address in (None, 0, ctypes.c_void_p(-1).value):
            result["error"] = f"mmap: {os.strerror(ctypes.get_errno())}"
            result["unknown_pages"] = pages
            address = None
            return result
        vector = (ctypes.c_ubyte * pages)()
        libc.mincore.restype = ctypes.c_int
        libc.mincore.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_ubyte),
        ]
        rc = libc.mincore(ctypes.c_void_p(address), length, vector)
        if rc != 0:
            result["error"] = f"mincore: {os.strerror(ctypes.get_errno())}"
            result["unknown_pages"] = pages
        else:
            resident = sum(1 for value in vector if value & 1)
            result.update(
                pages=pages,
                resident_pages=resident,
                nonresident_pages=pages - resident,
                unknown_pages=0,
            )
    except Exception as error:  # diagnostic helper must not masquerade as evidence
        result["error"] = str(error)
    finally:
        if address is not None:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            libc.munmap(ctypes.c_void_p(address), length)
        os.close(fd)
    return result


_HASH_CACHE = {}


def _sha256(path):
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def prepare_cache_condition(source, condition, copy_dir):
    """Prepare and verify an isolated model copy without global cache drops."""
    source = Path(source).resolve()
    meta = {
        "requested": condition,
        "actual": "uncontrolled",
        "controlled": False,
        "preparation_ok": False,
        "method": "none",
        "model": str(source),
        "source_size": None,
        "source_sha256": None,
        "copy_identity": None,
        "probe_before": None,
        "probe_after": None,
    }
    try:
        meta["source_size"] = source.stat().st_size
        meta["source_sha256"] = _sha256(source)
    except OSError as error:
        meta["error"] = str(error)
        return source, meta
    if condition == "uncontrolled":
        meta.update(
            actual="uncontrolled",
            controlled=False,
            preparation_ok=True,
            method="none",
            note="cache state was not controlled or verified before the run",
        )
        return source, meta
    try:
        copy_dir = Path(copy_dir)
        copy_dir.mkdir(parents=True, exist_ok=True)
        source_key = hashlib.sha256(str(source).encode()).hexdigest()[:10]
        key = f"{source.stem}-{source.stat().st_size}-{source_key}"
        destination = copy_dir / f"{key}{source.suffix}"
        if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
            shutil.copyfile(source, destination)
        meta["model"] = str(destination)
        meta["copy_identity"] = (
            f"{destination}:{destination.stat().st_size}:"
            f"source-sha256={meta['source_sha256']}"
        )
        meta["method"] = (
            "isolated copy + mincore verification (no cache drop)"
            if condition == "warm"
            else "isolated copy + POSIX_FADV_DONTNEED + mincore verification"
        )
        if condition == "warm":
            # Make the isolated copy genuinely warm before verifying it. Do not
            # assume that copy/writeback or a fresh process left pages resident.
            with destination.open("rb") as handle:
                for _ in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    pass
        meta["probe_before"] = _probe_residency(destination)
        if condition == "warm":
            probe = meta["probe_before"]
            pages = probe.get("pages") or 0
            resident = probe.get("resident_pages") or 0
            if pages > 0 and resident == pages:
                meta.update(
                    actual="warm",
                    controlled=False,
                    preparation_ok=True,
                    note="all probed pages were resident; this is a warm run, not cold",
                )
            else:
                meta.update(
                    actual="warm-or-unknown",
                    controlled=False,
                    preparation_ok=True,
                    note=f"{resident}/{pages} probed pages were resident; no cold claim is made",
                )
            return destination, meta
        fd = os.open(destination, os.O_RDONLY)
        try:
            # A freshly copied file may still be dirty. DONTNEED cannot evict
            # dirty pages, so make the copy clean before the hint.
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        probe = _probe_residency(destination)
        meta["probe_after"] = probe
        if probe.get("error"):
            meta.update(
                actual="unknown",
                preparation_ok=False,
                note=f"residency verification failed: {probe['error']}",
            )
            return destination, meta
        pages = probe.get("pages") or 0
        resident = probe.get("resident_pages") or 0
        nonresident = probe.get("nonresident_pages") or 0
        if pages > 0 and nonresident == pages:
            meta.update(
                actual="controlled-nonresident",
                controlled=True,
                preparation_ok=True,
                note="all probed pages were nonresident after isolated-copy preparation",
            )
        elif nonresident > 0:
            meta.update(
                actual="partially-controlled",
                controlled=False,
                preparation_ok=True,
                note=(
                    f"{nonresident}/{pages} probed pages nonresident; "
                    "the rest remained resident"
                ),
            )
        else:
            meta.update(
                actual="warm-or-unknown",
                controlled=False,
                preparation_ok=True,
                note=(
                    f"fadvise returned success but {resident}/{pages} probed pages "
                    "remained resident; no cold claim is made"
                ),
            )
    except Exception as error:
        meta.update(
            actual="cache-preparation-failed",
            controlled=False,
            preparation_ok=False,
            note=str(error),
        )
    return Path(meta["model"]), meta


def prepare_artifact_identity(path, model, directory, label):
    """Clone a profile/history for an identical isolated model copy.

    Profile validation intentionally binds to the model path and mtime. An
    isolated copy has a different path, so the benchmark rewrites only the
    identity fields. The tensor inventory and access records are unchanged.
    """
    if not path:
        return path
    source = Path(path)
    model = Path(model)
    try:
        payload = json.loads(source.read_text())
    except Exception:
        return path
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        return path
    stat = model.stat()
    if (
        identity.get("model_path") == str(model)
        and identity.get("model_size_bytes") == stat.st_size
        and identity.get("model_modified_unix_ns") == stat.st_mtime_ns
    ):
        return path
    identity["model_path"] = str(model)
    identity["model_size_bytes"] = stat.st_size
    identity["model_modified_unix_ns"] = stat.st_mtime_ns
    if isinstance(payload.get("profile"), dict):
        payload["profile"]["model_id"] = str(model)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{model.name}.{label}.json"
    destination.write_text(json.dumps(payload, sort_keys=True))
    return str(destination)


def command_for(
    binary, model, profile, history, target, ctx, prompt, tokens, arm, enforce, timeout, observe, observe_pages
):
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
    if observe:
        cmd += ["--residency-observe", "--residency-observe-pages", str(observe_pages)]
    return cmd


def parse_report(path, arm, workload, repeat, order, wall_ns, status, stderr):
    if status == 0 and Path(path).is_file():
        report = json.loads(Path(path).read_text())
        prediction = report.get("prediction") or {}
        metrics = prediction.get("metrics") or {}
        plan = prediction.get("plan") or {}
        staging = prediction.get("staging") or {}
        history = prediction.get("history") or {}
        observation = report.get("residency_observation") or {}
        page_before = observation.get("before") or {}
        page_after_staging = observation.get("after_staging") or {}
        correlation = observation.get("prefetch_correlation") or {}
        before = (report.get("measurements") or {}).get("before") or {}
        after = (report.get("measurements") or {}).get("after") or {}
        timing = report.get("timing") or {}
        execution_timing = report.get("execution_timing") or {}
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
            "end_to_end_ns": execution_timing.get("end_to_end_ns"),
            "inference_ns": execution_timing.get("inference_ns"),
            "profile_load_ns": execution_timing.get("profile_load_ns"),
            "history_load_ns": execution_timing.get("history_load_ns"),
            "prediction_planning_ns": execution_timing.get("prediction_ns"),
            "residency_planning_ns": execution_timing.get("planning_ns"),
            "staging_timing_ns": execution_timing.get("staging_ns"),
            "observation_timing_ns": execution_timing.get("observation_ns"),
            "synchronization_ns": execution_timing.get("synchronization_ns"),
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
            "observation_before_ns": page_before.get("elapsed_ns"),
            "observation_after_staging_ns": page_after_staging.get("elapsed_ns"),
            "observation_inspected_pages": page_before.get("inspected_pages"),
            "observation_sampled": page_before.get("sampled"),
            "observation_targeted": (page_before.get("targeted") or {}),
            "observation_targeted_pages_available": (page_before.get("targeted") or {}).get("targeted_pages_available"),
            "observation_targeted_pages_sampled": (page_before.get("targeted") or {}).get("targeted_pages_sampled"),
            "observation_general_pages_sampled": (page_before.get("targeted") or {}).get("general_pages_sampled"),
            "observation_selected_fully": (page_before.get("targeted") or {}).get("selected_resources_fully_observed"),
            "observation_selected_partial": (page_before.get("targeted") or {}).get("selected_resources_partially_observed"),
            "observation_selected_not_observed": (page_before.get("targeted") or {}).get("selected_resources_not_observed"),
            "resident_pages_before": page_before.get("resident_pages"),
            "resident_pages_after_staging": page_after_staging.get("resident_pages"),
            "unknown_pages_before": page_before.get("unknown_pages"),
            "prefetch_already_resident": correlation.get("already_resident"),
            "prefetch_became_resident": correlation.get("became_resident"),
            "prefetch_remained_nonresident": correlation.get("remained_nonresident"),
            "prefetch_correlation_unknown": correlation.get("unknown"),
            "prefetch_redundant": correlation.get("redundant"),
            "prefetch_potentially_useful": correlation.get("potentially_useful"),
            "prefetch_wasted": correlation.get("wasted"),
            "prefetch_ineffective": correlation.get("ineffective"),
            "prefetch_evicted_before_use": correlation.get("evicted_before_use"),
            "prefetch_before_access_observation_available": correlation.get("before_access_observation_available"),
            "prefetch_clearly_useful": correlation.get("clearly_useful"),
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
    parser.add_argument("--observe", action="store_true", help="enable bounded page residency observation")
    parser.add_argument("--observe-pages", type=int, default=2048)
    parser.add_argument(
        "--cache-condition",
        choices=["uncontrolled", "warm", "isolated-nonresident"],
        default="uncontrolled",
        help="safe cache preparation; warm is verified and never assumed",
    )
    parser.add_argument(
        "--model-copy-dir",
        default=os.environ.get(
            "RAMFORGE_COPY_DIR",
            str(Path.home() / ".cache" / "ramforge-cold-models"),
        ),
        help="directory for isolated model copies",
    )
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
                run_model, cache_meta = prepare_cache_condition(
                    args.model, args.cache_condition, args.model_copy_dir
                )
                prepared_profile = prepare_artifact_identity(
                    args.profile,
                    run_model,
                    Path(args.model_copy_dir) / "prepared",
                    "profile",
                )
                prepared_history = prepare_artifact_identity(
                    args.history,
                    run_model,
                    Path(args.model_copy_dir) / "prepared",
                    "history",
                )
                cmd = command_for(
                    args.binary,
                    str(run_model),
                    prepared_profile,
                    prepared_history,
                    args.target,
                    ctx,
                    prompt,
                    tokens,
                    arm,
                    args.enforce,
                    args.timeout,
                    args.observe,
                    args.observe_pages,
                )
                cmd += ["--residency-report-json", str(json_path)]
                start = time.time_ns()
                with err_path.open("w") as err:
                    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=err)
                wall = time.time_ns() - start
                row = parse_report(
                    json_path, arm, name, repeat, order, wall, proc.returncode, str(err_path)
                )
                row.update(
                    {
                        "cache_condition_requested": cache_meta["requested"],
                        "cache_condition_actual": cache_meta["actual"],
                        "cache_controlled": cache_meta["controlled"],
                        "cache_preparation_ok": cache_meta["preparation_ok"],
                        "cache_method": cache_meta["method"],
                        "cache_note": cache_meta.get("note"),
                        "model_file": cache_meta["model"],
                        "model_copy_identity": cache_meta["copy_identity"],
                        "model_source_sha256": cache_meta["source_sha256"],
                        "profile_prepared": prepared_profile != args.profile if args.profile else False,
                        "history_prepared": prepared_history != args.history if args.history else False,
                        "cache_probe_before": cache_meta.get("probe_before"),
                        "cache_probe_after": cache_meta.get("probe_after"),
                    }
                )
                rows.append(row)

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
        "cache_condition_requested": args.cache_condition,
        "cache_controlled": False,
        "cache_note": "Cache conditions are recorded per run; uncontrolled runs are not labeled cold.",
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
        "cache_conditions": {},
        "summary": {},
    }
    for row in rows:
        condition = row.get("cache_condition_actual") or "unknown"
        summary["cache_conditions"][condition] = summary["cache_conditions"].get(condition, 0) + 1
    summary["cache_controlled"] = bool(rows) and all(
        row.get("cache_controlled") for row in rows if row.get("status") == 0
    ) and not any(row.get("status") != 0 for row in rows)
    if args.cache_condition == "warm":
        summary["cache_note"] = "Warm runs are isolated-copy probes; no cold claim is made."

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
                "end_to_end_ns": spread(selected, "end_to_end_ns"),
                "inference_ns": spread(selected, "inference_ns"),
                "profile_load_ns": spread(selected, "profile_load_ns"),
                "history_load_ns": spread(selected, "history_load_ns"),
                "prediction_planning_ns": spread(selected, "prediction_planning_ns"),
                "residency_planning_ns": spread(selected, "residency_planning_ns"),
                "staging_timing_ns": spread(selected, "staging_timing_ns"),
                "observation_timing_ns": spread(selected, "observation_timing_ns"),
                "synchronization_ns": spread(selected, "synchronization_ns"),
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
                "observation_before_ns": spread(selected, "observation_before_ns"),
                "observation_after_staging_ns": spread(selected, "observation_after_staging_ns"),
                "observation_inspected_pages": median(selected, "observation_inspected_pages"),
                "observation_sampled": median(selected, "observation_sampled"),
                "observation_targeted_pages_available": median(selected, "observation_targeted_pages_available"),
                "observation_targeted_pages_sampled": median(selected, "observation_targeted_pages_sampled"),
                "observation_general_pages_sampled": median(selected, "observation_general_pages_sampled"),
                "observation_selected_fully": median(selected, "observation_selected_fully"),
                "observation_selected_partial": median(selected, "observation_selected_partial"),
                "observation_selected_not_observed": median(selected, "observation_selected_not_observed"),
                "resident_pages_before": median(selected, "resident_pages_before"),
                "resident_pages_after_staging": median(selected, "resident_pages_after_staging"),
                "prefetch_already_resident": median(selected, "prefetch_already_resident"),
                "prefetch_became_resident": median(selected, "prefetch_became_resident"),
                "prefetch_remained_nonresident": median(selected, "prefetch_remained_nonresident"),
                "prefetch_correlation_unknown": median(selected, "prefetch_correlation_unknown"),
                "prefetch_redundant": median(selected, "prefetch_redundant"),
                "prefetch_potentially_useful": median(selected, "prefetch_potentially_useful"),
                "prefetch_wasted": median(selected, "prefetch_wasted"),
                "prefetch_ineffective": median(selected, "prefetch_ineffective"),
                "prefetch_evicted_before_use": median(selected, "prefetch_evicted_before_use"),
                "prefetch_before_access_observation_available": median(selected, "prefetch_before_access_observation_available"),
                "prefetch_clearly_useful": median(selected, "prefetch_clearly_useful"),
                "prefetch_followed": median(selected, "prefetch_followed"),
                "prefetch_not_followed": median(selected, "prefetch_not_followed"),
                "prefetch_unknown": median(selected, "prefetch_unknown"),
            }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    lines = [
        "RAMforge prediction benchmark (simulation not included)",
        f"model={args.model}",
        f"profile={args.profile} history={args.history or 'none'}",
        f"target={args.target} enforce={args.enforce} observe={args.observe} observe_pages={args.observe_pages} repeats={args.repeats}",
        f"cache_controlled={summary['cache_controlled']} conditions={summary['cache_conditions']} ({summary['cache_note']})",
        f"runs={summary['runs']} failures={summary['failures']}",
    ]
    for workload in summary["workloads"]:
        lines.append(f"\n[{workload}]")
        for arm in ARMS:
            entry = summary["summary"][workload][arm]
            ttft = entry.get("ttft_ns") or {}
            major = entry.get("major_faults_delta") or {}
            lines.append(
                "  {arm}: end_to_end_median_ns={e2e} inference_median_ns={inference} "
                "ttft_median_ns={ttft} ttft_min={tmin} ttft_max={tmax} "
                "major_fault_delta_median={major} accepted_bytes={accepted} "
                "matched={matched} within_horizon={within} precision_bps={precision} "
                "recall_bps={recall} staging_latency_ns={staging}".format(
                    arm=arm,
                    e2e=(entry.get("end_to_end_ns") or {}).get("median"),
                    inference=(entry.get("inference_ns") or {}).get("median"),
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
