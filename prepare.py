# prepare.py
import argparse
import pickle
import json
from pathlib import Path

def _patch_pydantic_setstate():
    """
    Make unpickling older pydantic objects more tolerant.
    This fixes KeyError '__fields_set__' in many cases.
    """
    try:
        from pydantic.v1.main import BaseModel as PydanticBaseModel  # pydantic v2 compat
        orig = PydanticBaseModel.__setstate__

        def patched(self, state):
            if isinstance(state, dict) and "__fields_set__" not in state:
                state["__fields_set__"] = set()
            return orig(self, state)

        PydanticBaseModel.__setstate__ = patched
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stats", required=False)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    _patch_pydantic_setstate()

    with inp.open("rb") as f:
        data = pickle.load(f)

    # Convert to a stable, portable format: list[dict(page_content, metadata)]
    processed = []
    if isinstance(data, list):
        for x in data:
            if isinstance(x, dict) and "page_content" in x:
                page = x.get("page_content", "")
                meta = x.get("metadata", {}) or {}
            else:
                page = getattr(x, "page_content", str(x))
                meta = getattr(x, "metadata", {}) or {}
            processed.append({"page_content": page, "metadata": dict(meta)})
    else:
        processed = [{"page_content": str(data), "metadata": {}}]

    with out.open("wb") as f:
        pickle.dump(processed, f)

    if args.stats:
        stats_path = Path(args.stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        lens = [len(d["page_content"]) for d in processed] if processed else []
        stats = {
            "num_records": len(processed),
            "avg_chars": (sum(lens) / len(lens)) if lens else 0,
            "min_chars": min(lens) if lens else 0,
            "max_chars": max(lens) if lens else 0,
        }
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
