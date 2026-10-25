import unittest
from unittest.mock import patch

from beyond_similarity.retrieval import CodeNode
from beyond_similarity.semantic import SemanticRetriever


class SemanticTests(unittest.TestCase):
    def test_empty_nodes_rejected_before_model_load(self):
        # No model dependency should be required to validate empty input.
        with self.assertRaises((ValueError, RuntimeError)):
            SemanticRetriever([])


if __name__ == "__main__":
    unittest.main()
