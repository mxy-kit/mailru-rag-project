# tests/test_compile.py
import compileall
import pathlib

def test_python_files_compile():
    root = pathlib.Path(__file__).resolve().parents[1]  # repo root
    ok = compileall.compile_dir(
        str(root),
        quiet=1,
        maxlevels=50,
    )
    assert ok, "Some .py files failed to compile (syntax error?)"
