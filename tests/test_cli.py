import tempfile
import unittest
from pathlib import Path

from beyond_similarity.cli import load_tasks, run_evaluation, search_python


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
        from beyond_similarity.cli import evaluate_task
        task = {
            "query": "find parser",
            "nodes": [{"id": "parser", "name": "parse"}],
            "relevant_ids": ["not-a-node"],
        }
        with self.assertRaises(ValueError):
            evaluate_task(task, (1,))

    def test_search_python_returns_ranked_functions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "module.py").write_text(
                "def validate_input(value):\n    return value is not None\n\n"
                "def render_page(template):\n    return template\n",
                encoding="utf-8",
            )
            result = search_python(root, "validate input", 5, "graph", False, "")
            self.assertGreaterEqual(result["indexed_nodes"], 2)
            self.assertEqual(result["results"][0]["name"], "validate_input")


if __name__ == "__main__":
    unittest.main()
