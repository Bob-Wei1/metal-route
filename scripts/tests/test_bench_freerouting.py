#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "bench-freerouting.py"
SPEC = importlib.util.spec_from_file_location("bench_freerouting", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


METALROUTE_FAKE = r"""#!/usr/bin/env python3
import json
import os
import pathlib
import sys

trace = pathlib.Path(__file__).with_name("trace.jsonl")
with trace.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "tool": "metalroute",
        "argv": sys.argv[1:],
        "rayon": os.environ.get("RAYON_NUM_THREADS"),
        "metal_override": os.environ.get("METALROUTE_EXPERIMENTAL_METAL_ISOLATED"),
        "cell_budget": os.environ.get("MR_CELL_BUDGET"),
    }) + "\n")

if "--version" in sys.argv:
    print("metalroute 0.test")
    raise SystemExit(0)

dsn = pathlib.Path(sys.argv[sys.argv.index("--input") + 1])
if "INCOMPATIBLE" in dsn.read_text(encoding="utf-8"):
    print("failed to convert DSN to problem: unsupported DSN", file=sys.stderr)
    raise SystemExit(1)

if "--max-nets" in sys.argv:
    print("DSN parsed: layers=2 components=1 pads=3 nets=2 (skipped 0 <2-pin), board 1.00x1.00 mm, min_trace_width 0.200 mm", file=sys.stderr)
    print("RESULT route-dsn nets=0 routed=0 conn=0.0% vias=0 wall=0.001s grid=10x10x2L")
    raise SystemExit(0)

ses = pathlib.Path(sys.argv[sys.argv.index("--ses") + 1])
ses.write_text("(session metalroute)\n", encoding="utf-8")
print("RESULT route-dsn nets=2 routed=2 conn=100.0% vias=1 wall=0.010s grid=10x10x2L")
"""


JAVA_FAKE = r"""#!/usr/bin/env python3
import json
import pathlib
import sys

trace = pathlib.Path(__file__).with_name("trace.jsonl")
with trace.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({"tool": "java", "argv": sys.argv[1:]}) + "\n")

if "-version" in sys.argv:
    print("openjdk version fake", file=sys.stderr)
    raise SystemExit(0)
if "--help" in sys.argv:
    print("Freerouting v2.3.0 (build-date: test)")
    raise SystemExit(0)
if "-drc" in sys.argv:
    report = pathlib.Path(sys.argv[sys.argv.index("-drc") + 1])
    design = sys.argv[sys.argv.index("-de") + 1]
    post_reload = "+" in design
    violations = [{}] if post_reload and "metalroute" in design else []
    report.write_text(json.dumps({
        "unconnected_items": [] if post_reload else [{}, {}],
        "violations": violations,
        "quality_score": 1.0 if violations else 0.0,
    }), encoding="utf-8")
    raise SystemExit(0)
if "-do" in sys.argv:
    ses = pathlib.Path(sys.argv[sys.argv.index("-do") + 1])
    ses.write_text("(session freerouting)\n", encoding="utf-8")
    print("Auto-routing stage completed: started with 2 unrouted nets, completed in 0.01 seconds")
    raise SystemExit(0)
raise SystemExit(3)
"""


class HarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.metalroute = self.make_executable("metalroute", METALROUTE_FAKE)
        self.java = self.make_executable("java", JAVA_FAKE)
        self.jar = self.root / "freerouting-2.3.0.jar"
        self.jar.write_bytes(b"fake jar")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_executable(self, name: str, content: str) -> Path:
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def run_harness(self, dsn_text: str) -> tuple[int, dict]:
        dsn = self.root / "fixture.dsn"
        dsn.write_text(dsn_text, encoding="utf-8")
        output = self.root / "results"
        old_metal = os.environ.get("METALROUTE_EXPERIMENTAL_METAL_ISOLATED")
        old_budget = os.environ.get("MR_CELL_BUDGET")
        os.environ["METALROUTE_EXPERIMENTAL_METAL_ISOLATED"] = "1"
        os.environ["MR_CELL_BUDGET"] = "adaptive"
        try:
            with mock.patch.object(
                bench, "EXPECTED_FREEROUTING_SHA256", bench.sha256_file(self.jar)
            ):
                exit_code = bench.main(
                    [
                        "--metalroute",
                        str(self.metalroute),
                        "--freerouting-jar",
                        str(self.jar),
                        "--java",
                        str(self.java),
                        "--repetitions",
                        "2",
                        "--timeout",
                        "5",
                        "--output-dir",
                        str(output),
                        str(dsn),
                    ]
                )
        finally:
            if old_metal is None:
                os.environ.pop("METALROUTE_EXPERIMENTAL_METAL_ISOLATED", None)
            else:
                os.environ["METALROUTE_EXPERIMENTAL_METAL_ISOLATED"] = old_metal
            if old_budget is None:
                os.environ.pop("MR_CELL_BUDGET", None)
            else:
                os.environ["MR_CELL_BUDGET"] = old_budget
        return exit_code, json.loads(
            (output / "report.json").read_text(encoding="utf-8")
        )

    def test_complete_report_uses_same_input_one_worker_ses_and_reload(self) -> None:
        exit_code, report = self.run_harness("(pcb compatible)\n")
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["schema_version"], bench.SCHEMA_VERSION)
        fixture = report["fixtures"][0]
        self.assertEqual(fixture["workload_check"]["status"], "matched")
        self.assertEqual(fixture["comparison"]["status"], "complete")
        self.assertFalse(fixture["comparison"]["post_reload_quality_equal"])
        self.assertIsNone(fixture["comparison"]["equal_quality_speedup"])
        self.assertEqual(fixture["engines"]["metalroute"]["validated_runs"], 2)
        self.assertEqual(fixture["engines"]["freerouting"]["validated_runs"], 2)

        serialized = json.dumps(report)
        self.assertNotIn(str(self.root), serialized)
        trace = [
            json.loads(line)
            for line in (self.root / "trace.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        metal_routes = [
            item
            for item in trace
            if item["tool"] == "metalroute" and "--ses" in item["argv"]
        ]
        self.assertTrue(metal_routes)
        self.assertTrue(all(item["rayon"] == "1" for item in metal_routes))
        self.assertTrue(all(item["metal_override"] is None for item in metal_routes))
        self.assertTrue(all(item["cell_budget"] is None for item in metal_routes))
        self.assertTrue(all("--out" not in item["argv"] for item in metal_routes))
        self.assertTrue(all("--drc" not in item["argv"] for item in metal_routes))
        free_routes = [
            item for item in trace if item["tool"] == "java" and "-do" in item["argv"]
        ]
        self.assertTrue(free_routes)
        self.assertTrue(
            all("--router.max_threads=1" in item["argv"] for item in free_routes)
        )
        self.assertTrue(
            all("--router.max_passes=500" in item["argv"] for item in free_routes)
        )
        self.assertTrue(
            all(
                "--router.optimizer.enabled=true" in item["argv"]
                for item in free_routes
            )
        )
        self.assertTrue(all("-Xmx8g" in item["argv"] for item in free_routes))
        self.assertTrue(
            all(
                "-XX:+HeapDumpOnOutOfMemoryError" in item["argv"]
                for item in free_routes
            )
        )
        self.assertTrue(
            all("--router.job_timeout=00:00:05" in item["argv"] for item in free_routes)
        )
        self.assertTrue(
            all(
                "--router.fanout.timeout=00:15:00" in item["argv"]
                for item in free_routes
            )
        )
        self.assertTrue(
            all(
                "--router.optimizer.timeout=00:10:00" in item["argv"]
                for item in free_routes
            )
        )
        self.assertTrue(
            all("--logging.file.level=INFO" in item["argv"] for item in free_routes)
        )
        reloads = [
            item
            for item in trace
            if item["tool"] == "java"
            and "-drc" in item["argv"]
            and "+" in item["argv"][item["argv"].index("-de") + 1]
        ]
        self.assertEqual(len(reloads), 4)
        last_route_index = max(
            index
            for index, item in enumerate(trace)
            if "-do" in item["argv"] or "--ses" in item["argv"]
        )
        first_reload_index = min(trace.index(item) for item in reloads)
        self.assertGreater(first_reload_index, last_route_index)

    def test_incompatible_metalroute_input_withholds_ratio(self) -> None:
        exit_code, report = self.run_harness("(pcb INCOMPATIBLE)\n")
        self.assertEqual(exit_code, 2)
        fixture = report["fixtures"][0]
        self.assertEqual(
            fixture["input_compatibility"]["metalroute"]["status"], "incompatible"
        )
        self.assertEqual(
            fixture["comparison"]["status"], "input_incompatible_or_probe_error"
        )
        self.assertIsNone(fixture["comparison"]["wall_time_factor"])

    def test_schema_declares_pinned_version(self) -> None:
        schema_path = (
            SCRIPT.parents[1] / "benchmarks" / "freerouting" / "report.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(
            schema["properties"]["schema_version"]["const"], bench.SCHEMA_VERSION
        )
        version = schema["properties"]["tools"]["properties"]["freerouting"]["allOf"][
            1
        ]["properties"]["version"]["const"]
        self.assertEqual(version, bench.EXPECTED_FREEROUTING_VERSION)

    def test_official_fixture_directory_selects_labeled_smoke_set(self) -> None:
        fixture_dir = self.root / "DAC2020_boards"
        fixture_dir.mkdir()
        for filename in bench.OFFICIAL_SMOKE_FIXTURES:
            (fixture_dir / filename).write_text("(pcb fixture)\n", encoding="utf-8")
        args = bench.build_parser().parse_args(
            [
                "--metalroute",
                str(self.metalroute),
                "--freerouting-jar",
                str(self.jar),
                "--java",
                str(self.java),
                "--official-fixture-dir",
                str(fixture_dir),
            ]
        )
        expected = {
            filename: bench.sha256_file(fixture_dir / filename)
            for filename in bench.OFFICIAL_SMOKE_FIXTURES
        }
        with mock.patch.object(bench, "OFFICIAL_SMOKE_FIXTURE_SHA256", expected):
            _, _, dsns, official_smoke = bench.validate_inputs(args)
        self.assertTrue(official_smoke)
        self.assertEqual(
            [path.name for path in dsns], list(bench.OFFICIAL_SMOKE_FIXTURES)
        )

    def test_speed_ratio_requires_faster_engine_to_be_no_worse(self) -> None:
        compatibility = {
            "metalroute": {"status": "compatible"},
            "freerouting": {"status": "compatible"},
        }
        workload = {"status": "matched"}

        def engine(wall: float, unconnected: int, violations: int) -> dict:
            return {
                "status": "complete",
                "median_external_wall_seconds": wall,
                "post_reload_quality": [
                    {
                        "unconnected_items": unconnected,
                        "violations": violations,
                        "quality_score": 0.0,
                    }
                ],
            }

        faster_but_worse = bench.compare_engines(
            compatibility,
            workload,
            engine(1.0, 0, 2),
            engine(2.0, 0, 1),
        )
        self.assertEqual(faster_but_worse["faster_engine"], "metalroute")
        self.assertIsNone(faster_but_worse["wall_time_factor"])
        self.assertIsNone(faster_but_worse["quality_gated_speedup"])

        faster_and_better = bench.compare_engines(
            compatibility,
            workload,
            engine(2.0, 0, 2),
            engine(1.0, 0, 1),
        )
        self.assertEqual(
            faster_and_better["quality_gated_speedup"],
            {"engine": "freerouting", "factor": 2.0},
        )

        equal_counts_different_scores = bench.compare_engines(
            compatibility,
            workload,
            engine(1.0, 0, 1),
            engine(2.0, 0, 1),
        )
        self.assertTrue(equal_counts_different_scores["post_reload_quality_equal"])
        self.assertEqual(
            equal_counts_different_scores["equal_quality_speedup"],
            {"engine": "metalroute", "factor": 2.0},
        )

    def test_common_quality_stability_ignores_engine_specific_score_jitter(self) -> None:
        runs = []
        for run_number, score in enumerate((10.0, 11.0), start=1):
            runs.append(
                {
                    "status": "ok",
                    "external_wall_seconds": float(run_number),
                    "post_reload_drc": {
                        "unconnected_items": 1,
                        "violations": 2,
                        "quality_score": score,
                    },
                }
            )
        summary = bench.summarize_engine(runs, 2)
        self.assertTrue(summary["quality_stable_across_runs"])
        self.assertEqual(bench.quality_cell(summary), "1 / 2")


if __name__ == "__main__":
    unittest.main()
