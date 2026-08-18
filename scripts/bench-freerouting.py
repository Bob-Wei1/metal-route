#!/usr/bin/env python3
"""Reproducible, same-DSN speed comparison with Freerouting 2.3.0.

This harness deliberately keeps routing and validation separate:

* the timed region is one fresh external process from DSN load through SES write;
* router worker pools are capped at one;
* Freerouting 2.3.0 reloads every SES and produces the post-route DRC report;
* a wall-time ratio is published only when both parsers see the same initial
  two-point workload and every measured output reloads successfully.

It downloads nothing. The caller supplies a metalroute release binary, a pinned
Freerouting JAR, and one or more externally managed DSN fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


EXPECTED_FREEROUTING_VERSION = "2.3.0"
EXPECTED_FREEROUTING_SHA256 = (
    "3cf18d608437740bc497db6b8ef5888e2e60a08de0def20691d1bad0c0e0ee24"
)
SCHEMA_VERSION = "metalroute.freerouting-speed.v1"
OFFICIAL_SMOKE_FIXTURES = (
    "DAC2020_bm08.dsn",
    "DAC2020_bm06.dsn",
    "DAC2020_bm07.dsn",
)
OFFICIAL_SMOKE_FIXTURE_SHA256 = {
    "DAC2020_bm08.dsn": "5d3acaaac47c1851d439150e3b70751b85fe1e8b8afc55278f1487b692b32bc5",
    "DAC2020_bm06.dsn": "31f38102d90a1bb4b901d4ca8d1877eb41752281ffa9de9f53a3cf69ba5231e2",
    "DAC2020_bm07.dsn": "39d85afa3133caae9b274350183868ad1fce5a0c64e3d5c6874598a899007c85",
}
METALROUTE_ENV_OVERRIDES = (
    "METALROUTE_EXPERIMENTAL_METAL_ISOLATED",
    "MR_CELL_BUDGET",
)

RESULT_RE = re.compile(
    r"^RESULT route-dsn nets=(?P<nets>\d+) routed=(?P<routed>\d+) "
    r"conn=(?P<connectivity>[\d.]+)% vias=(?P<vias>\d+) "
    r"wall=(?P<wall>[\d.]+)s grid=(?P<w>\d+)x(?P<h>\d+)x(?P<layers>\d+)L$",
    re.MULTILINE,
)
PARSE_STATS_RE = re.compile(
    r"DSN parsed: layers=(?P<layers>\d+) components=(?P<components>\d+) "
    r"pads=(?P<pads>\d+) nets=(?P<nets>\d+) "
    r"\(skipped (?P<skipped>\d+) <2-pin\)"
)
FREEROUTING_VERSION_RE = re.compile(r"Freerouting v(?P<version>\d+\.\d+\.\d+)")

METALROUTE_INCOMPATIBILITY_MARKERS = (
    "failed to convert dsn to problem",
    "failed to parse dsn",
    "unsupported dsn",
    "unsupported dsn resolution",
    "missing (structure",
    "missing board boundary",
)
FREEROUTING_INCOMPATIBILITY_MARKERS = (
    "failed to load board",
    "couldn't read the input file",
    "couldn't load the input file",
    "cannot load board",
    "dsn file couldn't be loaded",
)


class BenchError(RuntimeError):
    """A harness/setup error, as distinct from one fixture being incompatible."""


@dataclass
class ProcessResult:
    exit_code: int | None
    wall_seconds: float
    timed_out: bool
    stdout: str
    stderr: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return slug or "fixture"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_process(
    argv: Sequence[str],
    *,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> ProcessResult:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            list(argv),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=env,
        )
        result = ProcessResult(
            exit_code=completed.returncode,
            wall_seconds=time.perf_counter() - started,
            timed_out=False,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        result = ProcessResult(
            exit_code=None,
            wall_seconds=time.perf_counter() - started,
            timed_out=True,
            stdout=stdout,
            stderr=stderr,
        )

    if stdout_path is not None:
        write_text(stdout_path, result.stdout)
    if stderr_path is not None:
        write_text(stderr_path, result.stderr)
    return result


def parse_metalroute_result(text: str) -> dict[str, Any] | None:
    match = RESULT_RE.search(text)
    if match is None:
        return None
    values = match.groupdict()
    return {
        "two_point_nets": int(values["nets"]),
        "routed_two_point_nets": int(values["routed"]),
        "connectivity_percent": float(values["connectivity"]),
        "vias": int(values["vias"]),
        "internal_router_wall_seconds": float(values["wall"]),
        "grid": {
            "width": int(values["w"]),
            "height": int(values["h"]),
            "layers": int(values["layers"]),
        },
    }


def parse_metalroute_stats(text: str) -> dict[str, int] | None:
    match = PARSE_STATS_RE.search(text)
    if match is None:
        return None
    values = match.groupdict()
    return {
        "layers": int(values["layers"]),
        "components": int(values["components"]),
        "pads": int(values["pads"]),
        "original_nets": int(values["nets"]),
        "nets_skipped_below_two_pins": int(values["skipped"]),
    }


def collection_count(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    raise ValueError(f"expected a JSON array or null, got {type(value).__name__}")


def public_log_path(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def error_excerpt(result: ProcessResult, limit: int = 1200) -> str | None:
    text = "\n".join(
        part.strip() for part in (result.stderr, result.stdout) if part.strip()
    )
    if not text:
        return None
    return text[-limit:]


def marker_found(result: ProcessResult, markers: Iterable[str]) -> bool:
    combined = f"{result.stdout}\n{result.stderr}".lower()
    return any(marker in combined for marker in markers)


def timeout_setting(seconds: float) -> str:
    whole = max(1, int(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def metalroute_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in METALROUTE_ENV_OVERRIDES:
        env.pop(name, None)
    env["RAYON_NUM_THREADS"] = "1"
    return env


def probe_tools(
    metalroute: Path,
    freerouting_jar: Path,
    java: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    freerouting_sha256 = sha256_file(freerouting_jar)
    if freerouting_sha256 != EXPECTED_FREEROUTING_SHA256:
        raise BenchError(
            "Freerouting JAR SHA-256 mismatch: expected "
            f"{EXPECTED_FREEROUTING_SHA256}, found {freerouting_sha256}"
        )
    metal = run_process([str(metalroute), "--version"], timeout_seconds=timeout_seconds)
    if metal.timed_out or metal.exit_code != 0:
        raise BenchError(
            f"metalroute --version failed: {error_excerpt(metal) or 'no output'}"
        )
    metal_version = metal.stdout.strip() or metal.stderr.strip()

    freerouting = run_process(
        [java, "-jar", str(freerouting_jar), "--help"],
        timeout_seconds=timeout_seconds,
    )
    version_match = FREEROUTING_VERSION_RE.search(
        f"{freerouting.stdout}\n{freerouting.stderr}"
    )
    if freerouting.timed_out or freerouting.exit_code != 0 or version_match is None:
        raise BenchError(
            "could not identify the Freerouting JAR version: "
            f"{error_excerpt(freerouting) or 'no version output'}"
        )
    version = version_match.group("version")
    if version != EXPECTED_FREEROUTING_VERSION:
        raise BenchError(
            f"expected Freerouting {EXPECTED_FREEROUTING_VERSION}, found {version}; "
            "use the pinned release JAR"
        )

    java_version_result = run_process(
        [java, "-version"], timeout_seconds=timeout_seconds
    )
    java_version = "\n".join(
        value.strip()
        for value in (java_version_result.stdout, java_version_result.stderr)
        if value.strip()
    )

    return {
        "metalroute": {
            "version_output": metal_version,
            "sha256": sha256_file(metalroute),
            "size_bytes": metalroute.stat().st_size,
        },
        "freerouting": {
            "version": version,
            "sha256": freerouting_sha256,
            "size_bytes": freerouting_jar.stat().st_size,
        },
        "java": {"version_output": java_version},
    }


def metalroute_preflight(
    metalroute: Path,
    dsn: Path,
    fixture_dir: Path,
    output_dir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    (fixture_dir / "preflight").mkdir(parents=True, exist_ok=True)
    stdout_path = fixture_dir / "preflight" / "metalroute.stdout.log"
    stderr_path = fixture_dir / "preflight" / "metalroute.stderr.log"
    env = metalroute_environment()
    result = run_process(
        [
            str(metalroute),
            "route-dsn",
            "--input",
            str(dsn),
            "--max-nets",
            "0",
        ],
        timeout_seconds=timeout_seconds,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    parsed = parse_metalroute_result(result.stdout)
    record: dict[str, Any] = {
        "status": "compatible",
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "wall_seconds": result.wall_seconds,
        "logs": {
            "stdout": public_log_path(stdout_path, output_dir),
            "stderr": public_log_path(stderr_path, output_dir),
        },
        "parse_stats": parse_metalroute_stats(result.stderr),
    }
    if result.timed_out:
        record["status"] = "probe_timeout"
        record["error"] = "zero-net DSN ingest probe timed out"
    elif result.exit_code != 0:
        record["status"] = (
            "incompatible"
            if marker_found(result, METALROUTE_INCOMPATIBILITY_MARKERS)
            else "probe_error"
        )
        record["error"] = "zero-net DSN ingest probe failed; see referenced logs"
    elif parsed is None or parsed["two_point_nets"] != 0:
        record["status"] = "probe_error"
        record["error"] = "metalroute did not emit the expected zero-net RESULT line"
    return record


def freerouting_drc(
    *,
    java: str,
    freerouting_jar: Path,
    dsn: Path,
    ses: Path | None,
    report_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    output_dir: Path,
    timeout_seconds: float,
) -> tuple[dict[str, Any], ProcessResult]:
    design_arg = str(dsn) if ses is None else f"{dsn}+{ses}"
    result = run_process(
        [
            java,
            "-jar",
            str(freerouting_jar),
            "-de",
            design_arg,
            "-drc",
            str(report_path),
            "--gui.enabled=false",
            "--api_server.enabled=false",
        ],
        timeout_seconds=timeout_seconds,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    record: dict[str, Any] = {
        "status": "ok",
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "wall_seconds": result.wall_seconds,
        "logs": {
            "stdout": public_log_path(stdout_path, output_dir),
            "stderr": public_log_path(stderr_path, output_dir),
        },
    }
    if result.timed_out:
        record["status"] = "timeout"
        record["error"] = "Freerouting DRC reload timed out"
        return record, result
    if result.exit_code != 0:
        record["status"] = "process_error"
        record["error"] = "Freerouting DRC process failed; see referenced logs"
        return record, result
    if not report_path.is_file():
        record["status"] = "missing_report"
        record["error"] = "Freerouting exited successfully without writing the DRC JSON"
        return record, result
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        record.update(
            {
                "unconnected_items": collection_count(payload.get("unconnected_items")),
                "violations": collection_count(payload.get("violations")),
                "quality_score": payload.get("quality_score"),
                "report_sha256": sha256_file(report_path),
                "report": public_log_path(report_path, output_dir),
            }
        )
    except (OSError, json.JSONDecodeError, ValueError):
        record["status"] = "malformed_report"
        record["error"] = (
            "could not parse Freerouting DRC JSON; see referenced report/logs"
        )
    return record, result


def freerouting_preflight(
    java: str,
    freerouting_jar: Path,
    dsn: Path,
    fixture_dir: Path,
    output_dir: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    preflight_dir = fixture_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    drc, process = freerouting_drc(
        java=java,
        freerouting_jar=freerouting_jar,
        dsn=dsn,
        ses=None,
        report_path=preflight_dir / "freerouting-baseline-drc.json",
        stdout_path=preflight_dir / "freerouting.stdout.log",
        stderr_path=preflight_dir / "freerouting.stderr.log",
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
    )
    if drc["status"] == "ok":
        return {"status": "compatible", "baseline_drc": drc}
    status = (
        "incompatible"
        if marker_found(process, FREEROUTING_INCOMPATIBILITY_MARKERS)
        else "probe_error"
    )
    if drc["status"] == "timeout":
        status = "probe_timeout"
    return {"status": status, "baseline_drc": drc, "error": drc.get("error")}


def metalroute_run(
    *,
    metalroute: Path,
    dsn: Path,
    run_number: int,
    fixture_dir: Path,
    output_dir: Path,
    timeout_seconds: float,
) -> tuple[dict[str, Any], Path | None]:
    run_dir = fixture_dir / "metalroute" / f"run-{run_number:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ses_path = run_dir / "routed.ses"
    stdout_path = run_dir / "route.stdout.log"
    stderr_path = run_dir / "route.stderr.log"
    env = metalroute_environment()
    result = run_process(
        [
            str(metalroute),
            "route-dsn",
            "--input",
            str(dsn),
            "--ses",
            str(ses_path),
        ],
        timeout_seconds=timeout_seconds,
        env=env,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    parsed = parse_metalroute_result(result.stdout)
    record: dict[str, Any] = {
        "run": run_number,
        "status": "route_ok",
        "route_status": "route_ok",
        "external_wall_seconds": result.wall_seconds,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "logs": {
            "stdout": public_log_path(stdout_path, output_dir),
            "stderr": public_log_path(stderr_path, output_dir),
        },
        "router_result": parsed,
    }
    if result.timed_out:
        record["status"] = record["route_status"] = "route_timeout"
        record["error"] = "metalroute timed out"
    elif result.exit_code != 0:
        record["status"] = record["route_status"] = "route_error"
        record["error"] = "metalroute process failed; see referenced logs"
    elif parsed is None:
        record["status"] = record["route_status"] = "malformed_result"
        record["error"] = "metalroute did not emit its RESULT line"
    elif not ses_path.is_file() or ses_path.stat().st_size == 0:
        record["status"] = record["route_status"] = "missing_ses"
        record["error"] = "metalroute exited successfully without a non-empty SES"
    else:
        record["ses"] = {
            "sha256": sha256_file(ses_path),
            "size_bytes": ses_path.stat().st_size,
            "path": public_log_path(ses_path, output_dir),
        }
        return record, ses_path
    return record, None


def freerouting_run(
    *,
    java: str,
    freerouting_jar: Path,
    dsn: Path,
    passes: int,
    run_number: int,
    fixture_dir: Path,
    output_dir: Path,
    timeout_seconds: float,
) -> tuple[dict[str, Any], Path | None]:
    run_dir = fixture_dir / "freerouting" / f"run-{run_number:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ses_path = run_dir / "routed.ses"
    freerouting_log_path = run_dir / "freerouting.log"
    stdout_path = run_dir / "route.stdout.log"
    stderr_path = run_dir / "route.stderr.log"
    result = run_process(
        [
            java,
            "-Dsun.stdout.buffered=false",
            "-Xmx8g",
            "-Xms256m",
            "-XX:+HeapDumpOnOutOfMemoryError",
            f"-XX:HeapDumpPath={run_dir}",
            "-jar",
            str(freerouting_jar),
            "-de",
            str(dsn),
            "-do",
            str(ses_path),
            f"--router.max_passes={passes}",
            "--router.max_threads=1",
            f"--router.job_timeout={timeout_setting(timeout_seconds)}",
            "--router.optimizer.enabled=true",
            "--router.fanout.enabled=true",
            "--router.enabled=true",
            "--router.fanout.timeout=00:15:00",
            "--router.optimizer.timeout=00:10:00",
            "--logging.file.level=INFO",
            f"--logging.file.location={freerouting_log_path}",
            "--logging.console.level=INFO",
            "--api_server.enabled=false",
            "--gui.enabled=false",
        ],
        timeout_seconds=timeout_seconds,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    record: dict[str, Any] = {
        "run": run_number,
        "status": "route_ok",
        "route_status": "route_ok",
        "external_wall_seconds": result.wall_seconds,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "logs": {
            "stdout": public_log_path(stdout_path, output_dir),
            "stderr": public_log_path(stderr_path, output_dir),
            "freerouting": public_log_path(freerouting_log_path, output_dir),
        },
    }
    if result.timed_out:
        record["status"] = record["route_status"] = "route_timeout"
        record["error"] = "Freerouting timed out"
    elif result.exit_code != 0:
        record["status"] = record["route_status"] = "route_error"
        record["error"] = "Freerouting process failed; see referenced logs"
    elif not ses_path.is_file() or ses_path.stat().st_size == 0:
        record["status"] = record["route_status"] = "missing_ses"
        record["error"] = "Freerouting exited successfully without a non-empty SES"
    else:
        record["ses"] = {
            "sha256": sha256_file(ses_path),
            "size_bytes": ses_path.stat().st_size,
            "path": public_log_path(ses_path, output_dir),
        }
        return record, ses_path
    return record, None


def add_post_reload_drc(
    *,
    engine: str,
    runs_and_paths: list[tuple[dict[str, Any], Path | None]],
    java: str,
    freerouting_jar: Path,
    dsn: Path,
    output_dir: Path,
    timeout_seconds: float,
) -> None:
    for record, ses_path in runs_and_paths:
        if ses_path is None:
            continue
        run_dir = ses_path.parent
        drc, _ = freerouting_drc(
            java=java,
            freerouting_jar=freerouting_jar,
            dsn=dsn,
            ses=ses_path,
            report_path=run_dir / "post-reload-drc.json",
            stdout_path=run_dir / "drc.stdout.log",
            stderr_path=run_dir / "drc.stderr.log",
            output_dir=output_dir,
            timeout_seconds=timeout_seconds,
        )
        record["post_reload_drc"] = drc
        if drc["status"] == "ok":
            record["status"] = "ok"
        else:
            record["status"] = "reload_error"
            record["error"] = (
                f"{engine} SES did not pass Freerouting reload: {drc.get('error')}"
            )


def summarize_engine(runs: list[dict[str, Any]], requested_runs: int) -> dict[str, Any]:
    validated = [run for run in runs if run["status"] == "ok"]
    wall_samples = [run["external_wall_seconds"] for run in validated]
    quality_tuples = sorted(
        {
            (
                run["post_reload_drc"]["unconnected_items"],
                run["post_reload_drc"]["violations"],
                run["post_reload_drc"].get("quality_score"),
            )
            for run in validated
        },
        key=lambda item: tuple(-1 if value is None else value for value in item),
    )
    common_quality_counts = {
        (item[0], item[1])
        for item in quality_tuples
    }
    return {
        "status": (
            "complete"
            if requested_runs > 0 and len(validated) == requested_runs
            else "incomplete"
        ),
        "requested_runs": requested_runs,
        "executed_runs": len(runs),
        "validated_runs": len(validated),
        "median_external_wall_seconds": (
            statistics.median(wall_samples) if wall_samples else None
        ),
        "external_wall_seconds": wall_samples,
        "post_reload_quality": [
            {
                "unconnected_items": item[0],
                "violations": item[1],
                "quality_score": item[2],
            }
            for item in quality_tuples
        ],
        # Freerouting's aggregate score is engine-specific diagnostics. The
        # comparison's shared quality contract is only U/V, so score jitter must
        # not make otherwise identical common quality look unstable.
        "quality_stable_across_runs": len(common_quality_counts) == 1,
        "runs": runs,
    }


def workload_check(
    metalroute_runs: list[dict[str, Any]], freerouting_preflight_record: dict[str, Any]
) -> dict[str, Any]:
    metal_counts = sorted(
        {
            run["router_result"]["two_point_nets"]
            for run in metalroute_runs
            if run.get("router_result") is not None
        }
    )
    baseline = freerouting_preflight_record.get("baseline_drc", {})
    freerouting_count = baseline.get("unconnected_items")
    record: dict[str, Any] = {
        "status": "unavailable",
        "metalroute_two_point_net_counts": metal_counts,
        "freerouting_initial_unconnected_items": freerouting_count,
        "note": (
            "Equality is a conservative cross-parser gate, not proof that every DSN "
            "rule or geometry feature has identical semantics."
        ),
    }
    if len(metal_counts) == 1 and isinstance(freerouting_count, int):
        record["status"] = (
            "matched" if metal_counts[0] == freerouting_count else "mismatched"
        )
    return record


def compare_engines(
    compatibility: dict[str, Any],
    workload: dict[str, Any],
    metalroute: dict[str, Any],
    freerouting: dict[str, Any],
) -> dict[str, Any]:
    if any(value["status"] != "compatible" for value in compatibility.values()):
        return {
            "status": "input_incompatible_or_probe_error",
            "faster_engine": None,
            "wall_time_factor": None,
            "post_reload_quality_equal": None,
            "quality_gated_speedup": None,
        }
    if workload["status"] != "matched":
        return {
            "status": "workload_mismatch_or_unavailable",
            "faster_engine": None,
            "wall_time_factor": None,
            "post_reload_quality_equal": None,
            "quality_gated_speedup": None,
        }
    metal_wall = metalroute["median_external_wall_seconds"]
    free_wall = freerouting["median_external_wall_seconds"]
    if (
        metalroute["status"] != "complete"
        or freerouting["status"] != "complete"
        or metal_wall is None
        or free_wall is None
        or metal_wall <= 0
        or free_wall <= 0
    ):
        return {
            "status": "incomplete_runs_or_reload",
            "faster_engine": None,
            "wall_time_factor": None,
            "post_reload_quality_equal": None,
            "quality_gated_speedup": None,
        }
    if metal_wall <= free_wall:
        faster = "metalroute"
        factor = free_wall / metal_wall
    else:
        faster = "freerouting"
        factor = metal_wall / free_wall

    metal_quality = metalroute["post_reload_quality"]
    free_quality = freerouting["post_reload_quality"]
    metal_counts = {
        (item["unconnected_items"], item["violations"])
        for item in metal_quality
    }
    free_counts = {
        (item["unconnected_items"], item["violations"])
        for item in free_quality
    }
    quality_equal = (
        len(metal_counts) == 1
        and len(free_counts) == 1
        and metal_counts == free_counts
    )
    faster_no_worse = False
    if len(metal_counts) == 1 and len(free_counts) == 1:
        metal_unconnected, metal_violations = next(iter(metal_counts))
        free_unconnected, free_violations = next(iter(free_counts))
        if faster == "metalroute":
            faster_no_worse = (
                metal_unconnected <= free_unconnected
                and metal_violations <= free_violations
            )
        else:
            faster_no_worse = (
                free_unconnected <= metal_unconnected
                and free_violations <= metal_violations
            )
    gated_speedup = (
        {"engine": faster, "factor": factor} if faster_no_worse else None
    )
    return {
        "status": "complete",
        "faster_engine": faster,
        "wall_time_factor": factor if faster_no_worse else None,
        "median_external_wall_seconds": {
            "metalroute": metal_wall,
            "freerouting": free_wall,
        },
        "post_reload_quality_equal": quality_equal,
        "quality_gated_speedup": gated_speedup,
        "equal_quality_speedup": (
            {"engine": faster, "factor": factor} if quality_equal else None
        ),
        "interpretation": (
            "The medians compare fresh-process end-to-end wall time. A factor is "
            "reported only when the faster engine is no worse in both post-reload "
            "unconnected-item and violation counts."
        ),
    }


def benchmark_fixture(
    *,
    dsn: Path,
    metalroute: Path,
    freerouting_jar: Path,
    java: str,
    repetitions: int,
    passes: int,
    timeout_seconds: float,
    output_dir: Path,
    official_smoke: bool,
) -> dict[str, Any]:
    fixture_hash = sha256_file(dsn)
    fixture_dir = output_dir / "artifacts" / f"{slugify(dsn.stem)}-{fixture_hash[:12]}"
    fixture_dir.mkdir(parents=True, exist_ok=True)

    metal_preflight = metalroute_preflight(
        metalroute, dsn, fixture_dir, output_dir, timeout_seconds
    )
    free_preflight = freerouting_preflight(
        java, freerouting_jar, dsn, fixture_dir, output_dir, timeout_seconds
    )
    compatibility = {
        "metalroute": metal_preflight,
        "freerouting": free_preflight,
    }

    metal_pairs: list[tuple[dict[str, Any], Path | None]] = []
    free_pairs: list[tuple[dict[str, Any], Path | None]] = []
    for run_number in range(1, repetitions + 1):
        # Alternate launch order to reduce systematic thermal/order bias.
        order = (
            ("metalroute", "freerouting")
            if run_number % 2
            else (
                "freerouting",
                "metalroute",
            )
        )
        for engine in order:
            if engine == "metalroute" and metal_preflight["status"] == "compatible":
                metal_pairs.append(
                    metalroute_run(
                        metalroute=metalroute,
                        dsn=dsn,
                        run_number=run_number,
                        fixture_dir=fixture_dir,
                        output_dir=output_dir,
                        timeout_seconds=timeout_seconds,
                    )
                )
            elif engine == "freerouting" and free_preflight["status"] == "compatible":
                free_pairs.append(
                    freerouting_run(
                        java=java,
                        freerouting_jar=freerouting_jar,
                        dsn=dsn,
                        passes=passes,
                        run_number=run_number,
                        fixture_dir=fixture_dir,
                        output_dir=output_dir,
                        timeout_seconds=timeout_seconds,
                    )
                )

    # Validation is intentionally outside every timed routing interval.
    add_post_reload_drc(
        engine="metalroute",
        runs_and_paths=metal_pairs,
        java=java,
        freerouting_jar=freerouting_jar,
        dsn=dsn,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
    )
    add_post_reload_drc(
        engine="freerouting",
        runs_and_paths=free_pairs,
        java=java,
        freerouting_jar=freerouting_jar,
        dsn=dsn,
        output_dir=output_dir,
        timeout_seconds=timeout_seconds,
    )

    metal_runs = [record for record, _ in metal_pairs]
    free_runs = [record for record, _ in free_pairs]
    workload = workload_check(metal_runs, free_preflight)
    metal_summary = summarize_engine(metal_runs, repetitions)
    free_summary = summarize_engine(free_runs, repetitions)
    comparison = compare_engines(compatibility, workload, metal_summary, free_summary)
    fixture_identity: dict[str, Any] = {
        "filename": dsn.name,
        "sha256": fixture_hash,
        "size_bytes": dsn.stat().st_size,
    }
    if official_smoke:
        fixture_identity["source"] = {
            "repository": "https://github.com/freerouting/freerouting",
            "tag": "v2.3.0",
            "path": f"scripts/benchmark/fixtures/DAC2020_boards/{dsn.name}",
        }
    return {
        "fixture": fixture_identity,
        "input_compatibility": compatibility,
        "workload_check": workload,
        "engines": {
            "metalroute": metal_summary,
            "freerouting": free_summary,
        },
        "comparison": comparison,
    }


def quality_cell(engine: dict[str, Any]) -> str:
    qualities = engine["post_reload_quality"]
    if not qualities:
        return "—"
    counts = {
        (item["unconnected_items"], item["violations"])
        for item in qualities
    }
    if len(counts) != 1:
        return "varies"
    unconnected, violations = next(iter(counts))
    return f"{unconnected} / {violations}"


def seconds_cell(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f} s"


def render_markdown(report: dict[str, Any]) -> str:
    method = report["methodology"]
    tools = report["tools"]
    lines = [
        "# metalroute vs Freerouting 2.3.0",
        "",
        f"Generated `{report['generated_at']}` from schema `{report['schema_version']}`.",
        "",
        "This is a same-input, fresh-process wall-time comparison. Each timed run "
        "starts at process launch, loads the DSN, routes, and writes an SES. The "
        "Freerouting DRC reload happens afterward and is excluded from timing.",
        "",
        f"- Repetitions: {method['repetitions']} (median reported)",
        f"- Fixture profile: {method['fixture_profile']['name']}",
        f"- Freerouting passes: {method['freerouting_max_passes']}",
        f"- External timeout: {method['external_timeout_seconds']} seconds per process",
        "- Router workers: one (`RAYON_NUM_THREADS=1`; `--router.max_threads=1`)",
        f"- metalroute: `{tools['metalroute']['version_output']}` "
        f"(`{tools['metalroute']['sha256'][:12]}…`)",
        f"- Freerouting: `{tools['freerouting']['version']}` "
        f"(`{tools['freerouting']['sha256'][:12]}…`)",
        "",
        "| Fixture | Gate | metalroute median | Freerouting median | Quality-gated ratio | metalroute DRC U / V | Freerouting DRC U / V |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for fixture in report["fixtures"]:
        comparison = fixture["comparison"]
        metal = fixture["engines"]["metalroute"]
        free = fixture["engines"]["freerouting"]
        speedup = comparison.get("quality_gated_speedup")
        factor = (
            f"{speedup['engine']} {speedup['factor']:.2f}×"
            if speedup is not None
            else "—"
        )
        lines.append(
            "| {name} | {gate} | {metal_time} | {free_time} | {factor} | {mq} | {fq} |".format(
                name=fixture["fixture"]["filename"].replace("|", "\\|"),
                gate=comparison["status"],
                metal_time=seconds_cell(metal["median_external_wall_seconds"]),
                free_time=seconds_cell(free["median_external_wall_seconds"]),
                factor=factor,
                mq=quality_cell(metal),
                fq=quality_cell(free),
            )
        )
    lines.extend(
        [
            "",
            "`U / V` means Freerouting post-reload unconnected items / violations. "
            "A ratio appears only when the faster engine is no worse in both common "
            "post-reload quality counts. Raw median times remain visible otherwise.",
            "",
            "A ratio is withheld when either input probe fails, the initial workload "
            "counts differ, a route fails, an SES is missing, or Freerouting cannot "
            "reload an output. See `report.json` and the referenced logs for the exact gate.",
            "",
        ]
    )
    return "\n".join(lines)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_text(temporary, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def checkpoint(report: dict[str, Any], output_dir: Path) -> None:
    atomic_write_json(output_dir / "report.json", report)
    write_text(output_dir / "report.md", render_markdown(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare metalroute with pinned Freerouting 2.3.0 on the same DSNs."
    )
    parser.add_argument("dsn", nargs="*", type=Path, help="external DSN fixture(s)")
    parser.add_argument(
        "--official-fixture-dir",
        type=Path,
        help=(
            "external Freerouting v2.3.0 DAC2020_boards directory; selects the "
            "bm08/bm06/bm07 smoke set"
        ),
    )
    parser.add_argument(
        "--metalroute",
        type=Path,
        default=Path("target/release/metalroute"),
        help="release metalroute binary (default: target/release/metalroute)",
    )
    parser.add_argument(
        "--freerouting-jar",
        type=Path,
        required=True,
        help="Freerouting 2.3.0 executable JAR (never downloaded or copied)",
    )
    parser.add_argument(
        "--java", default="java", help="Java executable (default: java)"
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--passes", type=int, default=500)
    parser.add_argument(
        "--timeout", type=float, default=1800.0, help="seconds per process"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="new result directory (default: benchmarks/runs/<UTC>-freerouting)",
    )
    return parser


def validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, list[Path], bool]:
    metalroute = args.metalroute.expanduser().resolve()
    freerouting_jar = args.freerouting_jar.expanduser().resolve()
    if args.official_fixture_dir is not None and args.dsn:
        raise BenchError(
            "use either positional DSNs or --official-fixture-dir, not both"
        )
    official_smoke = args.official_fixture_dir is not None
    if official_smoke:
        fixture_dir = args.official_fixture_dir.expanduser().resolve()
        dsns = [(fixture_dir / name).resolve() for name in OFFICIAL_SMOKE_FIXTURES]
    else:
        dsns = [path.expanduser().resolve() for path in args.dsn]
    if not dsns:
        raise BenchError("provide DSN paths or --official-fixture-dir")
    if not metalroute.is_file():
        raise BenchError(
            f"metalroute binary not found: {metalroute}; run cargo build --release -p mr-cli"
        )
    if not os.access(metalroute, os.X_OK):
        raise BenchError(f"metalroute binary is not executable: {metalroute}")
    if not freerouting_jar.is_file():
        raise BenchError(f"Freerouting JAR not found: {freerouting_jar}")
    for dsn in dsns:
        if not dsn.is_file():
            raise BenchError(f"DSN fixture not found: {dsn}")
        if dsn.suffix.lower() != ".dsn":
            raise BenchError(f"fixture must have a .dsn extension: {dsn}")
        if "+" in str(dsn):
            raise BenchError(
                f"DSN path cannot contain '+', which Freerouting reserves for DSN+SES reload: {dsn}"
            )
        if official_smoke:
            expected = OFFICIAL_SMOKE_FIXTURE_SHA256[dsn.name]
            actual = sha256_file(dsn)
            if actual != expected:
                raise BenchError(
                    f"official fixture SHA-256 mismatch for {dsn.name}: "
                    f"expected {expected}, found {actual}"
                )
    if len(set(dsns)) != len(dsns):
        raise BenchError("the DSN fixture list contains duplicate paths")
    artifact_keys = [(slugify(dsn.stem), sha256_file(dsn)) for dsn in dsns]
    if len(set(artifact_keys)) != len(artifact_keys):
        raise BenchError("two DSN fixtures would map to the same artifact directory")
    if args.repetitions < 1:
        raise BenchError("--repetitions must be at least 1")
    if args.passes < 1:
        raise BenchError("--passes must be at least 1")
    if args.timeout <= 0:
        raise BenchError("--timeout must be positive")
    return metalroute, freerouting_jar, dsns, official_smoke


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metalroute, freerouting_jar, dsns, official_smoke = validate_inputs(args)
        if args.output_dir is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
            output_dir = Path("benchmarks/runs") / f"{stamp}-freerouting"
        else:
            output_dir = args.output_dir.expanduser()
        output_dir = output_dir.resolve()
        if "+" in str(output_dir):
            raise BenchError(
                "--output-dir cannot contain '+', which Freerouting reserves for DSN+SES reload"
            )
        if output_dir.exists():
            raise BenchError(
                f"refusing to write into existing output directory {output_dir}"
            )
        output_dir.mkdir(parents=True)

        tools = probe_tools(
            metalroute, freerouting_jar, args.java, min(args.timeout, 60.0)
        )
        report: dict[str, Any] = {
            "$schema": (
                "https://raw.githubusercontent.com/Bob-Wei1/metal-route/main/"
                "benchmarks/freerouting/report.schema.json"
            ),
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now(),
            "methodology": {
                "timed_scope": "fresh process: DSN load through SES write",
                "excluded_from_timing": [
                    "input compatibility probes",
                    "post-reload DRC",
                ],
                "repetitions": args.repetitions,
                "statistic": "median",
                "warmups": 0,
                "launch_order": "alternating by repetition",
                "fixture_profile": {
                    "name": (
                        "Freerouting v2.3.0 DAC2020 smoke (bm08, bm06, bm07)"
                        if official_smoke
                        else "caller-supplied external DSNs"
                    ),
                    "files": [dsn.name for dsn in dsns],
                    "downloads_performed_by_harness": False,
                },
                "freerouting_max_passes": args.passes,
                "external_timeout_seconds": args.timeout,
                "freerouting_profile": {
                    "upstream": (
                        "https://github.com/freerouting/freerouting/blob/v2.3.0/"
                        "scripts/benchmark/run-benchmarks.ps1"
                    ),
                    "max_passes": args.passes,
                    "max_threads": 1,
                    "heap_initial": "256m",
                    "heap_max": "8g",
                    "job_timeout": timeout_setting(args.timeout),
                    "fanout_timeout": "00:15:00",
                    "optimizer_timeout": "00:10:00",
                    "fanout_enabled": True,
                    "router_enabled": True,
                    "optimizer_enabled": True,
                    "file_log_level": "INFO",
                    "matches_official_defaults": (
                        args.passes == 500 and args.timeout == 1800.0
                    ),
                },
                "router_worker_policy": {
                    "metalroute": "RAYON_NUM_THREADS=1",
                    "freerouting": "--router.max_threads=1",
                    "scope_note": "caps router worker pools, not every auxiliary runtime thread",
                },
                "metalroute_environment_overrides_cleared": list(
                    METALROUTE_ENV_OVERRIDES
                ),
                "output_format": "Specctra SES",
                "validation": (
                    "Freerouting 2.3.0 reloads INPUT.dsn+OUTPUT.ses and writes JSON DRC"
                ),
                "comparison_gate": (
                    "both input probes pass; metalroute two-point nets equal Freerouting "
                    "baseline unconnected items; every timed SES reloads"
                ),
                "command_templates": {
                    "metalroute": (
                        "METALROUTE_EXPERIMENTAL_METAL_ISOLATED=<unset> "
                        "MR_CELL_BUDGET=<unset> RAYON_NUM_THREADS=1 $METALROUTE "
                        "route-dsn --input $DSN --ses $SES"
                    ),
                    "freerouting": (
                        "java -Dsun.stdout.buffered=false -Xmx8g -Xms256m "
                        "-XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=$RUN_DIR "
                        "-jar $FREEROUTING_JAR -de $DSN -do $SES "
                        "--router.max_passes=N --router.max_threads=1 "
                        "--router.job_timeout=HH:MM:SS "
                        "--router.optimizer.enabled=true --router.fanout.enabled=true "
                        "--router.enabled=true --router.fanout.timeout=00:15:00 "
                        "--router.optimizer.timeout=00:10:00 "
                        "--logging.file.level=INFO --logging.file.location=$LOG "
                        "--logging.console.level=INFO --api_server.enabled=false "
                        "--gui.enabled=false"
                    ),
                    "post_reload_drc": (
                        "java -jar $FREEROUTING_JAR -de $DSN+$SES -drc $REPORT "
                        "--gui.enabled=false --api_server.enabled=false"
                    ),
                },
            },
            "system": {
                "os": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "python": platform.python_version(),
            },
            "tools": tools,
            "fixtures": [],
        }
        checkpoint(report, output_dir)

        for index, dsn in enumerate(dsns, start=1):
            print(f"[{index}/{len(dsns)}] {dsn.name}", flush=True)
            fixture = benchmark_fixture(
                dsn=dsn,
                metalroute=metalroute,
                freerouting_jar=freerouting_jar,
                java=args.java,
                repetitions=args.repetitions,
                passes=args.passes,
                timeout_seconds=args.timeout,
                output_dir=output_dir,
                official_smoke=official_smoke,
            )
            report["fixtures"].append(fixture)
            checkpoint(report, output_dir)
            print(f"  gate: {fixture['comparison']['status']}", flush=True)

        print(f"JSON: {output_dir / 'report.json'}")
        print(f"Markdown: {output_dir / 'report.md'}")
        return (
            0
            if all(
                fixture["comparison"]["status"] == "complete"
                for fixture in report["fixtures"]
            )
            else 2
        )
    except (BenchError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
