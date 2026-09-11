"""Disk IO with metadata discipline.

Activations are saved as .npz plus a sibling .meta.json. The loader REFUSES
to return arrays whose metadata contradicts the config you are running under.
This is the single cheapest guard against the most expensive class of bug in
this project: silently analysing activations from the wrong model, the wrong
hook point, or the wrong token position.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: Iterable[Dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Activation storage
# ----------------------------------------------------------------------------

META_KEYS_MUST_MATCH = ("model_name", "hook_name", "position", "dtype")


def save_activations(
    acts: np.ndarray,
    labels: np.ndarray,
    entities: np.ndarray,
    meta: Dict[str, Any],
    path: str | Path,
) -> Path:
    """Save an activation tensor with its labels, entity ids and metadata.

    acts      : (n_layers, n_examples, d_model) float32
    labels    : (n_examples,) int
    entities  : (n_examples,) str  -- used for entity-disjoint splitting
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if acts.ndim != 3:
        raise ValueError(f"expected (n_layers, n_examples, d_model), got {acts.shape}")
    if len(labels) != acts.shape[1] or len(entities) != acts.shape[1]:
        raise ValueError("labels/entities length must match acts.shape[1]")

    np.savez_compressed(
        path,
        acts=acts.astype(np.float32),
        labels=np.asarray(labels),
        entities=np.asarray(entities, dtype=object).astype(str),
    )
    full_meta = dict(meta)
    full_meta.update(
        {
            "shape": list(acts.shape),
            "n_layers": int(acts.shape[0]),
            "n_examples": int(acts.shape[1]),
            "d_model": int(acts.shape[2]),
            "dtype": "float32",
            "sha256": sha256_file(path),
        }
    )
    write_json(full_meta, str(path) + ".meta.json")
    return path


def load_activations(path: str | Path, expect: Dict[str, Any] | None = None):
    """Load activations, refusing to return them if metadata contradicts `expect`.

    Returns (acts, labels, entities, meta).
    """
    path = Path(path)
    meta_path = Path(str(path) + ".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} missing. Activations without metadata are unusable; re-extract."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    if expect:
        mismatches = [
            (k, meta.get(k), expect[k])
            for k in META_KEYS_MUST_MATCH
            if k in expect and meta.get(k) != expect[k]
        ]
        if mismatches:
            lines = "\n".join(f"  {k}: file={got!r} config={want!r}" for k, got, want in mismatches)
            raise ValueError(
                "Activation cache does not match the current config:\n"
                + lines
                + "\nRe-extract rather than proceeding."
            )

    with np.load(path, allow_pickle=False) as z:
        acts = z["acts"]
        labels = z["labels"]
        entities = z["entities"]
    return acts, labels, entities, meta


def append_metrics(run_dir: str | Path, record: Dict[str, Any], fname: str = "metrics.jsonl") -> Path:
    """Append one metrics record. One line per evaluation, never overwritten."""
    p = Path(run_dir) / fname
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return p
