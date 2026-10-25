import unittest
from pathlib import Path

from beyond_similarity.cli import evaluate_task, load_tasks, run_evaluation


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def test_bundled_data_loads(self):
        tasks = load_tasks(ROOT / "data" / "example_tasks.json")
        self.assertEqual(len(tasks), 2)

    def test_evaluation_summary_has_both_methods(self):
        report = run_evaluation(ROOT / "data" / "example_tasks.json", (1, 3))
        self.assertEqual(report["summary"]["task_count"], 2)
        self.assertIn("lexical", report["summary"]["methods"])
        self.assertIn("graph_guided", report["summary"]["methods"])
        for method in report["summary"]["methods"].values():
            self.assertIn("mrr", method)
            self.assertIn("latency_ms", method)


if __name__ == "__main__":
    unittest.main()
