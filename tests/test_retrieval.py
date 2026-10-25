import unittest

from beyond_similarity.metrics import precision_at_k, recall_at_k, reciprocal_rank
from beyond_similarity.retrieval import CodeNode, RepositoryIndex, tokenize


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = RepositoryIndex(
            [
                CodeNode("parser", "parse_input", "Parse data and call the validator"),
                CodeNode("validator", "validate_input", "Reject malformed input"),
                CodeNode("renderer", "render_page", "Render page templates"),
            ],
            [("parser", "validator")],
        )

    def test_tokenizer_handles_identifiers_and_case(self):
        self.assertEqual(tokenize("Parse_Input value42"), ["parse_input", "value42"])

    def test_lexical_results_cover_all_nodes(self):
        results = self.index.rank_lexical("reject malformed input")
        self.assertEqual({result.node_id for result in results}, {"parser", "validator", "renderer"})
        self.assertEqual(results[0].node_id, "validator")

    def test_graph_proximity(self):
        scores = self.index.graph_proximity(["parser"], max_depth=1)
        self.assertEqual(scores["parser"], 1.0)
        self.assertEqual(scores["validator"], 0.5)
        self.assertEqual(scores["renderer"], 0.0)

    def test_graph_guided_ranking_is_deterministic(self):
        first = self.index.rank_graph_guided("reject malformed input")
        second = self.index.rank_graph_guided("reject malformed input")
        self.assertEqual([r.node_id for r in first], [r.node_id for r in second])

    def test_unknown_edge_is_rejected(self):
        with self.assertRaises(ValueError):
            RepositoryIndex([CodeNode("known", "known")], [("known", "missing")])

    def test_metrics(self):
        ranked = ["a", "b", "c"]
        relevant = {"b", "c"}
        self.assertAlmostEqual(recall_at_k(ranked, relevant, 2), 0.5)
        self.assertAlmostEqual(precision_at_k(ranked, relevant, 2), 0.5)
        self.assertAlmostEqual(reciprocal_rank(ranked, relevant), 0.5)

    def test_empty_query_returns_zero_lexical_scores(self):
        self.assertEqual(set(self.index.lexical_scores("")), set(self.index.nodes))
        self.assertTrue(all(score == 0.0 for score in self.index.lexical_scores("").values()))


if __name__ == "__main__":
    unittest.main()
