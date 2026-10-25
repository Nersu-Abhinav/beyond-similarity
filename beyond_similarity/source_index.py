"""Extract searchable Python functions/classes and approximate call dependencies.

This is a static-analysis convenience index, not a complete language server.
Dynamic dispatch, decorators, imports, and attribute calls can make call edges
incomplete. Review generated edges before treating them as ground truth.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SourceNode:
    id: str
    name: str
    text: str
    file: str
    kind: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class SourceIndex:
    nodes: list[SourceNode]
    edges: list[tuple[str, str]]
    skipped_files: list[dict[str, str]]

    def to_task_nodes(self) -> list[dict[str, str]]:
        return [
            {"id": node.id, "name": node.name, "text": node.text}
            for node in self.nodes
        ]

    def to_dict(self) -> dict:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [list(edge) for edge in self.edges],
            "skipped_files": self.skipped_files,
        }


def _stable_id(relative_path: str, qualified_name: str, source: str) -> str:
    digest = hashlib.sha1(
        f"{relative_path}\0{qualified_name}\0{source}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{relative_path}:{qualified_name}:{digest}"


def index_python_repository(
    root: str | Path,
    *,
    exclude_dirs: Iterable[str] = (
        ".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist"
    ),
    max_file_bytes: int = 2_000_000,
) -> SourceIndex:
    """Index Python functions and classes under root; infer same-file call edges."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(str(root_path))
    excluded = set(exclude_dirs)
    nodes: list[SourceNode] = []
    skipped: list[dict[str, str]] = []
    file_asts: list[tuple[str, ast.AST, dict[str, str]]] = []

    for path in sorted(root_path.rglob("*.py")):
        relative = path.relative_to(root_path)
        if excluded.intersection(relative.parts):
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                skipped.append({"file": str(relative), "reason": "file exceeds size limit"})
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(relative))
        except (OSError, UnicodeError, SyntaxError) as exc:
            skipped.append({"file": str(relative), "reason": str(exc)})
            continue

        source_lines = source.splitlines()
        definitions: dict[str, str] = {}
        parent_stack: list[str] = []

        class Collector(ast.NodeVisitor):
            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                qualified = ".".join(parent_stack + [node.name])
                snippet = ast.get_source_segment(source, node) or node.name
                node_id = _stable_id(str(relative), qualified, snippet)
                definitions[qualified] = node_id
                nodes.append(SourceNode(
                    id=node_id, name=qualified,
                    text=(ast.get_docstring(node) or "") + "\n" + snippet[:12000],
                    file=str(relative), kind="class",
                    start_line=node.lineno, end_line=getattr(node, "end_lineno", node.lineno),
                ))
                parent_stack.append(node.name)
                for child in node.body:
                    self.visit(child)
                parent_stack.pop()

            def _visit_function(self, node: ast.AST) -> None:
                name = getattr(node, "name", "<anonymous>")
                qualified = ".".join(parent_stack + [name])
                snippet = ast.get_source_segment(source, node) or name
                node_id = _stable_id(str(relative), qualified, snippet)
                definitions[qualified] = node_id
                definitions.setdefault(name, node_id)
                nodes.append(SourceNode(
                    id=node_id, name=qualified,
                    text=(ast.get_docstring(node) or "") + "\n" + snippet[:12000],
                    file=str(relative), kind="method" if parent_stack else "function",
                    start_line=node.lineno, end_line=getattr(node, "end_lineno", node.lineno),
                ))
                parent_stack.append(name)
                for child in getattr(node, "body", []):
                    self.visit(child)
                parent_stack.pop()

            visit_FunctionDef = _visit_function
            visit_AsyncFunctionDef = _visit_function

        Collector().visit(tree)
        file_asts.append((str(relative), tree, definitions))

    edges: set[tuple[str, str]] = set()
    for relative, tree, definitions in file_asts:
        # Calls are mapped to definitions in the same file by name. This is
        # deliberately conservative; cross-file/import resolution is future work.
        class Calls(ast.NodeVisitor):
            def __init__(self) -> None:
                self.current: list[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_function(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._visit_function(node)

            def _visit_function(self, node: ast.AST) -> None:
                name = getattr(node, "name", "<anonymous>")
                qualified = ".".join(self.current + [name])
                source_id = definitions.get(qualified) or definitions.get(name)
                if source_id:
                    self.current.append(name)
                    for child in getattr(node, "body", []):
                        self.visit(child)
                    self.current.pop()

            def visit_Call(self, node: ast.Call) -> None:
                if self.current:
                    caller_name = ".".join(self.current)
                    caller_id = definitions.get(caller_name) or definitions.get(self.current[-1])
                    func = node.func
                    target_name = func.id if isinstance(func, ast.Name) else (
                        func.attr if isinstance(func, ast.Attribute) else None
                    )
                    target_id = definitions.get(target_name) if target_name else None
                    if caller_id and target_id and caller_id != target_id:
                        edges.add(tuple(sorted((caller_id, target_id))))
                self.generic_visit(node)

        Calls().visit(tree)

    nodes.sort(key=lambda node: (node.file, node.start_line, node.name))
    return SourceIndex(nodes=nodes, edges=sorted(edges), skipped_files=skipped)
