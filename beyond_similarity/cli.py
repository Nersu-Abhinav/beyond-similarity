"""CLI for the Beyond Similarity retrieval prototype."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .metrics import mean, precision_at_k, recall_at_k, reciprocal_rank
from .retrieval import CodeNode, RepositoryIndex, rank_ids


def load_tasks(path: Path) -> list[dict[str, Any]]:
    """Load either a JSON array or JSONL task file."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    tasks = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            tasks.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
    return tasks


def evaluate_task(task: dict[str, Any], k_values: tuple[int, ...]) -> dict[str, Any]:
    """Evaluate lexical and graph-guided retrieval on one annotated task."""
    required = {"query", "nodes"}
    missing = required - task.keys()
    if missing:
        raise ValueError(f"task is missing required keys: {sorted(missing)}")
    if not isinstance(task["nodes"], list) or not task["nodes"]:
        raise ValueError("task nodes must be a non-empty list")

    nodes = [
        CodeNode(
            id=str(item["id"]),
            name=str(item.get("name", item["id"])),
            text=str(item.get("text", "")),
        )
        for item in task["nodes"]
    ]
    edges = [(str(edge[0]), str(edge[1])) for edge in task.get("edges", [])]
    relevant = [str(item) for item in task.get("relevant_ids", [])]
    unknown_relevant = set(relevant) - {node.id for node in nodes}
    if unknown_relevant:
        raise ValueError(f"relevant_ids contain unknown node IDs: {sorted(unknown_relevant)}")

    index = RepositoryIndex(nodes, edges)
    query = str(task["query"])

    outputs: dict[str, Any] = {"task_id": task.get("task_id", "unknown")}
    for method_name, method in (
        ("lexical", lambda: index.rank_lexical(query)),
        ("graph_guided", lambda: index.rank_graph_guided(query)),
    ):
        start = time.perf_counter()
        ranked = method()
        latency_ms = (time.perf_counter() - start) * 1000.0
        ids = rank_ids(ranked)
        record: dict[str, Any] = {
            "mrr": reciprocal_rank(ids, relevant),
            "latency_ms": latency_ms,
            "top_candidates": [
                {"id": item.node_id, "score": round(item.score, 6)}
                for item in ranked[:10]
            ],
        }
        for k in k_values:
            record[f"recall@{k}"] = recall_at_k(ids, relevant, k)
            record[f"precision@{k}"] = precision_at_k(ids, relevant, k)
        outputs[method_name] = record
    return outputs


def run_evaluation(path: Path, k_values: tuple[int, ...]) -> dict[str, Any]:
    """Return per-task results and macro-averaged metrics."""
    tasks = load_tasks(path)
    if not tasks:
        raise ValueError("dataset contains no tasks")
    results = [evaluate_task(task, k_values) for task in tasks]

    summary: dict[str, Any] = {"task_count": len(results), "methods": {}}
    for method_name in ("lexical", "graph_guided"):
        method_results = [item[method_name] for item in results]
        metric_names = ["mrr", "latency_ms"]
        for k in k_values:
            metric_names += [f"recall@{k}", f"precision@{k}"]
        summary["methods"][method_name] = {
            name: mean([float(item[name]) for item in method_results])
            for name in metric_names
        }
    return {
        "summary": summary,
        "per_task": results,
        "warning": (
            "If the input is synthetic or illustrative, these numbers are smoke-test "
            "outputs only and must not be reported as benchmark evidence."
        ),
    }


def demo() -> dict[str, Any]:
    """Run a tiny synthetic smoke test; it is not a benchmark."""
    fixture = {
        "task_id": "synthetic-smoke-test",
        "query": "reject invalid input during parsing",
        "nodes": [
            {"id": "parser", "name": "parse_input", "text": "Parse input data and call validation"},
            {"id": "validator", "name": "validate_input", "text": "Reject invalid or malformed input"},
            {"id": "renderer", "name": "render_page", "text": "Render a page template"},
            {"id": "logger", "name": "write_log", "text": "Write diagnostics to a log file"},
        ],
        "edges": [["parser", "validator"], ["renderer", "logger"]],
        "relevant_ids": ["validator"],
    }
    result = evaluate_task(fixture, (1, 3))
    result["note"] = "Synthetic smoke test only; not benchmark evidence."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-guided code retrieval prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="run a synthetic smoke test")
    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a JSON/JSONL task file")
    evaluate_parser.add_argument("--data", required=True, type=Path)
    evaluate_parser.add_argument("--k", nargs="+", default=[1, 5, 10], type=int)
    evaluate_parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "demo":
        result = demo()
    else:
        if any(k < 1 for k in args.k):
            parser.error("all k values must be >= 1")
        result = run_evaluation(args.data, tuple(sorted(set(args.k))))

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if getattr(args, "output", None):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
