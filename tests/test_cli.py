import unittest
from pathlib import Path

from beyond_similarity.cli import load_tasks, run_evaluation


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
        self.assertIn("warning", report)
        for method in report["summary"]["methods"].values():
            self.assertIn("mrr", method)
            self.assertIn("latency_ms", method)

    def test_unknown_relevant_id_is_rejected(self):
        task = {
            "query": "find parser",
            "nodes": [{"id": "parser", "name": "parse"}],
            "relevant_ids": ["not-a-node"],
        }
        from beyond_similarity.cli import evaluate_task
        with self.assertRaises(ValueError):
            evaluate_task(task, (1,))


if __name__ == "__main__":
    unittest.main()
