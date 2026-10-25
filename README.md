# Beyond Similarity: Graph-Guided Retrieval for AI Coding Agents

A reproducible research prototype investigating whether repository structure improves issue-to-code localization beyond lexical and semantic similarity alone.

> **Research status:** This repository provides a runnable baseline and graph-aware retrieval prototype. It does **not** claim benchmark improvements. Run the evaluation and report measured results before making empirical claims in a paper.

## Research question

Does combining lexical relevance, semantic similarity, and dependency-graph evidence rank relevant code components more effectively than simpler retrieval baselines?

## What is implemented

- A dependency-free lexical retrieval baseline (TF-IDF-like token overlap with IDF weighting).
- A graph-aware reranker that incorporates normalized graph proximity.
- Bounded graph expansion around retrieved candidates.
- Evaluation metrics: Recall@k, reciprocal rank / MRR, precision@k, and latency.
- A small synthetic fixture for smoke tests only; it is **not** evidence of real-world performance.
- JSONL input/output so you can adapt the prototype to a repository dataset.

## Quick start

Requires Python 3.10+; no third-party packages are needed.

```bash
python --version
python -m unittest discover -s tests -v
python -m beyond_similarity.cli demo
```

Run retrieval on a JSON file:

```bash
python -m beyond_similarity.cli evaluate --data data/example_tasks.json
```

The bundled example dataset is illustrative and synthetic. Replace it with properly licensed benchmark tasks before drawing research conclusions.

## Data format

Each task is a JSON object:

```json
{
  "task_id": "example-001",
  "query": "Handle invalid input in the data parser",
  "nodes": [
    {"id": "parser", "name": "parse_input", "text": "Parse and validate incoming input"},
    {"id": "validator", "name": "validate_input", "text": "Reject missing or malformed values"}
  ],
  "edges": [["parser", "validator"]],
  "relevant_ids": ["validator"]
}
```

- `nodes`: source-code components with stable IDs, names, and searchable text.
- `edges`: directed or undirected dependency links; the current prototype treats them as undirected for proximity scoring.
- `relevant_ids`: known relevant components used **only for evaluation**, never for ranking.

## Methods

1. **Lexical baseline:** IDF-weighted token overlap between issue text and node name/text.
2. **Graph-aware method:** lexical candidate scores combined with bounded graph proximity to initial candidates.
3. **Evaluation:** Recall@k, Precision@k, MRR, and per-task latency.

The current implementation intentionally does not pretend to provide neural embeddings. A semantic-embedding baseline can be added when a fixed embedding model and reproducible dependency/runtime are available. Score weights and expansion depth should be selected on development data and then frozen for the test split.

## Research protocol

For a credible paper:
1. Use the competition's permitted data and comply with its terms.
2. Define repository- or task-level train/development/test splits before tuning.
3. Compare lexical, embedding-only, graph-only, hybrid, and proposed variants on identical tasks and candidate budgets.
4. Perform component ablations and measure latency.
5. Report sample sizes, exact metrics, uncertainty, failures, and limitations.
6. Do not report the synthetic demo results as scientific evidence.

## Project structure

```text
beyond-similarity/
├── beyond_similarity/
│   ├── __init__.py
│   ├── retrieval.py
│   ├── metrics.py
│   └── cli.py
├── data/
│   └── example_tasks.json
├── tests/
│   └── test_retrieval.py
├── .gitignore
├── LICENSE
└── README.md
```

## Limitations

This is a small research prototype, not a production coding agent. It uses lexical token scoring and graph proximity, not a trained neural retriever. It does not execute code, generate patches, or guarantee that retrieved locations contain a defect. Graph quality, tokenization, candidate budgets, data leakage, and repository diversity can materially affect results.

## Citation

If this work becomes a paper, replace this section with the final title, authors, version, and DOI if one is issued.

## License

See [LICENSE](LICENSE). Check all dataset and benchmark licenses separately; this software license does not change the terms of third-party data.
