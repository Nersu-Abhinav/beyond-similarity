import tempfile
import unittest
from pathlib import Path

from beyond_similarity.source_index import index_python_repository


class SourceIndexTests(unittest.TestCase):
    def test_indexes_functions_and_same_file_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.py").write_text(
                'def validate(value):\n    """Check value."""\n    return value is not None\n\n'
                'def parse(value):\n    return validate(value)\n',
                encoding="utf-8",
            )
            result = index_python_repository(root)
            names = {node.name for node in result.nodes}
            self.assertIn("validate", names)
            self.assertIn("parse", names)
            self.assertEqual(len(result.edges), 1)

    def test_reports_syntax_errors_without_aborting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "broken.py").write_text("def nope(:\n", encoding="utf-8")
            result = index_python_repository(root)
            self.assertEqual(result.nodes, [])
            self.assertEqual(len(result.skipped_files), 1)


if __name__ == "__main__":
    unittest.main()
