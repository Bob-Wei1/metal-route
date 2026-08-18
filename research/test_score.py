import contextlib
import io
import unittest

from research import score


def make_report(boards, include_drc=True):
    """Build a small, internally consistent CorpusReport-shaped dictionary."""
    per_board = []
    for corpus, name, nets_total, nets_routed, drc in boards:
        board = {
            "corpus": corpus,
            "board": name,
            "nets_total": nets_total,
            "nets_routed": nets_routed,
            "error": None,
        }
        if include_drc:
            board["drc_violations"] = drc
        per_board.append(board)

    groups = []
    for corpus in sorted({board[0] for board in boards}):
        members = [board for board in per_board if board["corpus"] == corpus]
        nets_total = sum(board["nets_total"] for board in members)
        nets_routed = sum(board["nets_routed"] for board in members)
        groups.append(
            {
                "name": corpus,
                "boards": len(members),
                "nets_total": nets_total,
                "nets_routed": nets_routed,
                "completion_rate": nets_routed / nets_total,
                "fully_routed_boards": sum(
                    board["nets_total"] > 0
                    and board["nets_routed"] == board["nets_total"]
                    for board in members
                ),
            }
        )

    nets_total = sum(board["nets_total"] for board in per_board)
    nets_routed = sum(board["nets_routed"] for board in per_board)
    return {
        "router": "negotiated",
        "boards": len(per_board),
        "nets_total": nets_total,
        "nets_routed": nets_routed,
        "completion_rate": nets_routed / nets_total,
        "fully_routed_boards": sum(
            board["nets_total"] > 0
            and board["nets_routed"] == board["nets_total"]
            for board in per_board
        ),
        "groups": groups,
        "per_board": per_board,
    }


class CompareTests(unittest.TestCase):
    def compare(self, base, candidate):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = score.compare(base, candidate)
        return result, output.getvalue()

    def test_keeps_real_improvement_with_same_workload_and_no_quality_regression(self):
        base = make_report(
            [("bug-reports", "a", 10, 8, 2), ("srj15", "b", 10, 10, 0)]
        )
        candidate = make_report(
            [("bug-reports", "a", 10, 9, 1), ("srj15", "b", 10, 10, 0)]
        )

        result, output = self.compare(base, candidate)

        self.assertEqual(result, 0)
        self.assertIn("VERDICT: KEEP", output)
        self.assertIn("drc: violations 2 -> 1 (-1)", output)

    def test_rejects_same_count_with_different_board_or_corpus_identity(self):
        base = make_report([("bug-reports", "a", 10, 8, 0)])
        candidate = make_report([("srj15", "a", 10, 9, 0)])

        result, output = self.compare(base, candidate)

        self.assertEqual(result, 1)
        self.assertIn("missing boards: bug-reports/a", output)
        self.assertIn("unexpected boards: srj15/a", output)

    def test_rejects_duplicate_board_identity(self):
        base = make_report([("bug-reports", "a", 10, 8, 0)])
        candidate = make_report(
            [("bug-reports", "a", 10, 9, 0), ("bug-reports", "b", 10, 10, 0)]
        )
        candidate["per_board"][1]["board"] = "a"

        result, output = self.compare(base, candidate)

        self.assertEqual(result, 1)
        self.assertIn("invalid candidate report: duplicate board identity", output)

    def test_rejects_forged_top_level_quality_aggregates(self):
        base = make_report([("bug-reports", "a", 10, 8, 0)])

        mutations = {
            "nets_routed": 9,
            "completion_rate": 0.9,
            "fully_routed_boards": 1,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                candidate = make_report([("bug-reports", "a", 10, 8, 0)])
                candidate[field] = value

                result, output = self.compare(base, candidate)

                self.assertEqual(result, 1)
                self.assertIn("invalid candidate report:", output)
                self.assertIn(field, output)

    def test_rejects_forged_group_quality_aggregates(self):
        base = make_report([("bug-reports", "a", 10, 8, 0)])

        mutations = {
            "boards": 2,
            "nets_total": 11,
            "nets_routed": 9,
            "completion_rate": 0.9,
            "fully_routed_boards": 1,
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                candidate = make_report([("bug-reports", "a", 10, 8, 0)])
                candidate["groups"][0][field] = value

                result, output = self.compare(base, candidate)

                self.assertEqual(result, 1)
                self.assertIn("invalid candidate report:", output)
                self.assertIn(field, output)

    def test_rejects_per_board_nets_total_change(self):
        base = make_report([("bug-reports", "a", 10, 8, 0)])
        candidate = make_report([("bug-reports", "a", 11, 10, 0)])

        result, output = self.compare(base, candidate)

        self.assertEqual(result, 1)
        self.assertIn("per-board nets_total changed: bug-reports/a (10 -> 11)", output)

    def test_rejects_completion_regression(self):
        base = make_report([("bug-reports", "a", 10, 9, 0)])
        candidate = make_report([("bug-reports", "a", 10, 8, 0)])

        result, output = self.compare(base, candidate)

        self.assertEqual(result, 1)
        self.assertIn("completion not improved", output)

    def test_rejects_fully_routed_regression_despite_completion_gain(self):
        base = make_report(
            [("bug-reports", "a", 10, 10, 0), ("bug-reports", "b", 100, 0, 0)]
        )
        candidate = make_report(
            [("bug-reports", "a", 10, 9, 0), ("bug-reports", "b", 100, 20, 0)]
        )

        result, output = self.compare(base, candidate)

        self.assertEqual(result, 1)
        self.assertIn("fully_routed_boards dropped (0 < 1)", output)

    def test_rejects_drc_violation_and_clean_board_regression(self):
        base = make_report(
            [("bug-reports", "a", 10, 8, 0), ("bug-reports", "b", 10, 10, 0)]
        )
        candidate = make_report(
            [("bug-reports", "a", 10, 9, 1), ("bug-reports", "b", 10, 10, 0)]
        )

        result, output = self.compare(base, candidate)

        self.assertEqual(result, 1)
        self.assertIn("DRC violations increased (+1)", output)
        self.assertIn("DRC-clean boards dropped (-1)", output)

    def test_rejects_fully_routed_clean_regression_when_other_drc_totals_hold(self):
        base = make_report(
            [("bug-reports", "a", 10, 10, 0), ("bug-reports", "b", 100, 0, 1)]
        )
        candidate = make_report(
            [("bug-reports", "a", 10, 9, 0), ("bug-reports", "b", 100, 100, 1)]
        )

        result, output = self.compare(base, candidate)

        self.assertEqual(result, 1)
        self.assertIn("fully-routed DRC-clean boards dropped (-1)", output)

    def test_rejects_new_error_even_when_zero_drc_would_look_better(self):
        base = make_report(
            [("bug-reports", "a", 10, 8, 4), ("bug-reports", "b", 10, 8, 0)]
        )
        candidate = make_report(
            [("bug-reports", "a", 10, 9, 0), ("bug-reports", "b", 10, 8, 0)]
        )
        candidate["per_board"][0]["error"] = "router failed"

        result, output = self.compare(base, candidate)

        self.assertEqual(result, 1)
        self.assertIn("new board errors: bug-reports/a", output)

    def test_rejects_mixed_legacy_and_current_drc_schemas(self):
        base = make_report([("bug-reports", "a", 10, 8, 0)], include_drc=False)
        candidate = make_report([("bug-reports", "a", 10, 9, 0)])

        result, output = self.compare(base, candidate)

        self.assertEqual(result, 1)
        self.assertIn("DRC schema differs between baseline and candidate", output)

    def test_preserves_legacy_comparison_when_both_reports_lack_drc(self):
        base = make_report([("bug-reports", "a", 10, 8, 0)], include_drc=False)
        candidate = make_report([("bug-reports", "a", 10, 9, 0)], include_drc=False)

        result, output = self.compare(base, candidate)

        self.assertEqual(result, 0)
        self.assertIn("drc: legacy reports (gate unavailable)", output)
        self.assertIn("VERDICT: KEEP", output)


if __name__ == "__main__":
    unittest.main()
