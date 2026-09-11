"""Config loading, run identity, and provenance.

The rule this module enforces: every artefact you write to disk carries the
config that produced it and the git commit you were on. When you look at a
CSV in three weeks and cannot remember which sweep it came from, this is
what saves you.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load a YAML config, optionally merged with a base config it names.

    A config may contain `_base: configs/base.yaml`; keys in the child win.
    Overrides (e.g. from argparse) win over everything and are recorded.
    """
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    base_name = cfg.pop("_base", None)
    if base_name:
        base = load_config(base_name)
        cfg = _deep_merge(base, cfg)

    if overrides:
        cfg = _deep_merge(cfg, {k: v for k, v in overrides.items() if v is not None})
    cfg["_config_path"] = str(path.relative_to(REPO_ROOT)) if str(path).startswith(str(REPO_ROOT)) else str(path)
    return cfg


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def config_hash(cfg: Dict[str, Any], length: int = 8) -> str:
    """Stable short hash of a config, ignoring bookkeeping keys."""
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:length]


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip() + ("-dirty" if dirty else "")
    except Exception:
        return "no-git"


def make_run_id(cfg: Dict[str, Any], tag: str = "run") -> str:
    """A run id that is unique, sortable, and traceable back to a config."""
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{tag}_{config_hash(cfg)}"


def run_dir(run_id: str, create: bool = True) -> Path:
    d = REPO_ROOT / "results" / "runs" / run_id
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def save_provenance(run_id: str, cfg: Dict[str, Any], extra: Dict[str, Any] | None = None) -> Path:
    """Write config + commit + library versions next to the run's outputs."""
    d = run_dir(run_id)
    payload = {
        "run_id": run_id,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "config_path": cfg.get("_config_path"),
        "versions": library_versions(),
    }
    if extra:
        payload["extra"] = extra
    p = d / "provenance.json"
    with open(p, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return p


def library_versions() -> Dict[str, str]:
    names = [
        "numpy", "pandas", "sklearn", "scipy", "matplotlib",
        "torch", "transformers", "transformer_lens", "sae_lens", "datasets",
    ]
    out: Dict[str, str] = {}
    for n in names:
        try:
            mod = __import__(n)
            out[n] = str(getattr(mod, "__version__", "unknown"))
        except Exception:
            out[n] = "not-installed"
    return out


@dataclasses.dataclass
class Paths:
    """Canonical locations. Never hard-code a path anywhere else."""

    root: Path = REPO_ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def meta(self) -> Path:
        return self.data / "meta"

    @property
    def tables(self) -> Path:
        return self.root / "results" / "tables"

    @property
    def figures(self) -> Path:
        return self.root / "results" / "figures"

    @property
    def activations(self) -> Path:
        return self.root / "results" / "activations"

    def ensure(self) -> "Paths":
        for p in [self.processed, self.meta, self.tables, self.figures, self.activations]:
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths()
