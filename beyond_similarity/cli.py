"""CLI for the Beyond Similarity retrieval prototype."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .metrics import mean, precision_at_k, recall_at_k, reciprocal_rank
from .retrieval import CodeNode, RepositoryIndex, rank_ids
from .source_index import index_python_repository


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
            "Synthetic or illustrative input produces smoke-test outputs only, not benchmark evidence."
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


def search_python(root: Path, query: str, k: int, method: str, use_semantic: bool, model_name: str) -> dict[str, Any]:
    """Index a local Python repository and rank code components for a query."""
    if k < 1:
        raise ValueError("k must be >= 1")
    source_index = index_python_repository(root)
    if not source_index.nodes:
        raise ValueError(f"No Python functions/classes found under {root}")
    code_nodes = [
        CodeNode(id=node.id, name=node.name, text=node.text)
        for node in source_index.nodes
    ]
    index = RepositoryIndex(code_nodes, source_index.edges)
    lexical = index.lexical_scores(query)
    ranked = index.rank_graph_guided(query) if method == "graph" else index.rank_lexical(query)
    semantic_scores: dict[str, float] = {}
    if use_semantic:
        from .semantic import SemanticRetriever
        semantic = SemanticRetriever(code_nodes, model_name=model_name)
        semantic_scores = {
            item.node_id: max(0.0, min(1.0, (item.score + 1.0) / 2.0))
            for item in semantic.search(query, k=len(code_nodes))
        }
        graph_scores = {item.node_id: item.graph_score for item in ranked}
        scores = {
            node.id: 0.45 * lexical[node.id]
            + 0.35 * semantic_scores.get(node.id, 0.0)
            + 0.20 * graph_scores.get(node.id, 0.0)
            for node in code_nodes
        }
    else:
        scores = {item.node_id: item.score for item in ranked}
    metadata = {node.id: node for node in source_index.nodes}
    ordered = sorted(scores, key=lambda node_id: (-scores[node_id], node_id))[:k]
    return {
        "query": query,
        "method": "semantic+lexical+graph" if use_semantic else method,
        "model": model_name if use_semantic else None,
        "warning": "Static AST/call edges are approximate. Retrieval is not proof of a bug.",
        "indexed_nodes": len(code_nodes),
        "indexed_edges": len(source_index.edges),
        "skipped_files": source_index.skipped_files,
        "results": [
            {
                "id": node_id,
                "name": metadata[node_id].name,
                "file": metadata[node_id].file,
                "lines": [metadata[node_id].start_line, metadata[node_id].end_line],
                "score": round(scores[node_id], 6),
                "lexical_score": round(lexical[node_id], 6),
                "semantic_score": round(semantic_scores.get(node_id, 0.0), 6) if use_semantic else None,
                "source_preview": metadata[node_id].text[:500],
            }
            for node_id in ordered
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Graph-guided code retrieval prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("demo", help="run a synthetic smoke test")
    index_parser = subparsers.add_parser("index-python", help="index Python functions/classes into JSON")
    index_parser.add_argument("--root", type=Path, default=Path("."))
    index_parser.add_argument("--output", type=Path, default=Path("source-index.json"))
    search_parser = subparsers.add_parser("search-python", help="search a local Python repository")
    search_parser.add_argument("--root", type=Path, default=Path("."))
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--k", type=int, default=10)
    search_parser.add_argument("--method", choices=["lexical", "graph"], default="graph")
    search_parser.add_argument("--semantic", action="store_true", help="include optional embedding scores")
    search_parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a JSON/JSONL task file")
    evaluate_parser.add_argument("--data", required=True, type=Path)
    evaluate_parser.add_argument("--k", nargs="+", default=[1, 5, 10], type=int)
    evaluate_parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.command == "demo":
        result = demo()
    elif args.command == "index-python":
        source_index = index_python_repository(args.root)
        result = source_index.to_dict()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({
            "output": str(args.output),
            "nodes": len(source_index.nodes),
            "edges": len(source_index.edges),
            "skipped_files": len(source_index.skipped_files),
        }, indent=2))
        return
    elif args.command == "search-python":
        try:
            result = search_python(args.root, args.query, args.k, args.method, args.semantic, args.model)
        except (ValueError, NotADirectoryError) as exc:
            parser.error(str(exc))
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
