import ast
from pathlib import Path
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


def _violations(path):
    """Top-level statements that would run the analysis when the module is imported."""
    bad = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.For, ast.While, ast.With, ast.Try)):
            bad.append(f"line {node.lineno}: top-level {type(node).__name__}")
        elif isinstance(node, ast.If):
            if "__name__" not in ast.unparse(node.test):
                bad.append(f"line {node.lineno}: top-level if")
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = ast.unparse(node.value.func)
            if name in ("log", "print"):
                bad.append(f"line {node.lineno}: top-level {name}(...)")
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            name = ast.unparse(node.value.func)
            if name.split(".")[-1].startswith(("read_", "load_")):
                bad.append(f"line {node.lineno}: top-level {name}(...)")
    return bad


@pytest.mark.parametrize("path", sorted(SRC.glob("*.py")), ids=lambda p: p.name)
def test_importing_a_module_does_not_run_the_analysis(path):
    assert _violations(path) == []