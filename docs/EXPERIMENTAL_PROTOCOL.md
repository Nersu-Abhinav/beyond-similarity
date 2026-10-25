# Experimental Protocol

This document defines the minimum protocol for making defensible claims about Beyond Similarity.

## Research question

Does graph-guided retrieval improve issue-to-code localization over lexical and semantic baselines, and what is the cost in latency and complexity?

## Method variants

1. **Lexical:** the repository's IDF-weighted token baseline.
2. **Semantic:** a fixed embedding model, with exact model revision recorded.
3. **Graph-only:** a clearly documented candidate-generation and ranking rule.
4. **Lexical + semantic:** a calibrated fusion of lexical and embedding scores.
5. **Graph-guided hybrid:** lexical/semantic retrieval followed by bounded graph reranking.
6. **Ablations:** no graph expansion, alternate depth, alternate seed budget, and alternate graph weight.

The current CLI directly compares lexical and graph-guided lexical retrieval. Semantic and graph-only comparisons require a separate integration/evaluation path and must not be claimed as implemented baselines until added and run.

## Leakage controls

- Split by repository or task group when possible.
- Keep relevant IDs and test labels out of retrieval-time features.
- Tune score weights, graph depth, and candidate budgets only on development data.
- Freeze the selected configuration before final test evaluation.
- Use the same tasks and comparable candidate budgets for all methods.
- Record data licenses, filtering, duplicate handling, and preprocessing.

## Metrics

Report Recall@1/5/10, Precision@1/5/10, MRR, per-query latency, task count, and runtime environment. If the benchmark executes proposed patches, report its patch pass rate separately. Localization metrics do not establish patch correctness.

## Results template

Fill this table only with measured values:

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | Precision@5 | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Lexical | — | — | — | — | — | — |
| Semantic | — | — | — | — | — | — |
| Graph-only | — | — | — | — | — | — |
| Lexical + semantic | — | — | — | — | — | — |
| Graph-guided hybrid | — | — | — | — | — | — |

Dashes are unfilled placeholders, not zero scores. Do not report synthetic fixture metrics as benchmark results.

## Reproducibility checklist

- [ ] Dataset version, license, and task IDs recorded
- [ ] Repository-level train/dev/test split fixed
- [ ] Model identifier and revision pinned
- [ ] Retrieval parameters tuned on development data only
- [ ] Exact command and Git commit recorded
- [ ] Baselines run on identical tasks
- [ ] Per-task outputs and failures retained
- [ ] Confidence intervals or paired significance analysis considered
- [ ] Results independently reproduced

## Known limitations

The AST indexer currently handles Python syntax and approximates same-file call relationships. Dynamic dispatch, imported symbols, decorators, aliases, and attribute calls can cause missing or inaccurate edges. The graph is treated as undirected for proximity scoring. The optional MiniLM encoder is a general-purpose text model, not a specialized code encoder. These constraints must be acknowledged in any paper.
