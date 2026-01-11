import pathlib
import py_compile

def test_python_files_compile():
    root = pathlib.Path(__file__).resolve().parents[1]
    for p in root.rglob("*.py"):
        if any(x in p.parts for x in (".venv", "__pycache__")):
            continue
        py_compile.compile(str(p), doraise=True)
