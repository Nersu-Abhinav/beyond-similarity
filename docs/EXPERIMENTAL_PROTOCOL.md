# Experimental Protocol

This protocol is intended to prevent accidental overclaiming.

## Before evaluating

- Check the competition rules and dataset license.
- Freeze task splits at the repository/project level where possible.
- Do not use ground-truth relevant IDs when retrieving candidates.
- Tune graph weights, expansion depth, and seed count on development data only.
- Freeze parameters before final test evaluation.

## Required baselines

1. Lexical retrieval.
2. A fixed semantic embedding retriever, when the model and dependencies can be pinned.
3. Graph-only retrieval with a clearly defined candidate-generation rule.
4. Lexical + semantic hybrid.
5. Proposed graph-guided method.

Use the same query set and comparable candidate budgets. Report each method's limitations.

## Metrics

Report Recall@1/5/10, Precision@1/5/10, MRR, latency, task count, and runtime environment. If the benchmark supports executed patches, separately report the prescribed patch-validation metric. Localization metrics must not be described as patch correctness.

## Ablations

Disable graph expansion, vary graph depth, vary seed limit, and compare confidence-aware retrieval with fixed-budget expansion. Select parameters using development data, not test results.

## Results template

Fill this table only with measured values:

| Method | Recall@1 | Recall@5 | Recall@10 | MRR | Precision@5 | Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Lexical | — | — | — | — | — | — |
| Semantic embedding | — | — | — | — | — | — |
| Graph-only | — | — | — | — | — | — |
| Lexical + semantic | — | — | — | — | — | — |
| Proposed | — | — | — | — | — | — |

Do not interpret dashes as zero. They are placeholders until measurements have been collected.

## Reporting

State the number of evaluated tasks, the split strategy, the code/embedding versions, exact parameters, missing-data handling, and uncertainty where appropriate. Retain failed cases and discuss possible leakage, repository overlap, graph incompleteness, and other limitations.
