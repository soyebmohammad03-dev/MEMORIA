"""Structural promises the documentation makes: a dependency-free core and lazy extras."""

import ast
import pkgutil
import subprocess
import sys
from pathlib import Path

import memoria

EXTRAS = {"neural", "vectors"}  # the only modules that need an optional runtime
BLOCKED = ("numpy", "onnxruntime", "tokenizers", "faiss")


def test_core_imports_nothing_from_memoria() -> None:
    tree = ast.parse((Path(memoria.__file__).parent / "core.py").read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("memoria")
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith("memoria") for a in node.names)


def test_every_module_but_the_extras_imports_without_optional_runtimes() -> None:
    names = [m.name for m in pkgutil.iter_modules(memoria.__path__) if m.name not in EXTRAS]
    code = (
        "import importlib, sys\n"
        f"for b in {BLOCKED!r}: sys.modules[b] = None\n"
        f"for n in {names!r}: importlib.import_module('memoria.' + n)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
