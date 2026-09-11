"""Shared bootstrap for every script: path setup, args, dataset loading."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import PATHS, load_config, make_run_id, run_dir, save_provenance  # noqa: E402
from src.utils.io import read_jsonl, write_json  # noqa: E402
from src.utils.seed import set_all_seeds  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, default=None, help="override cfg.seed")
    ap.add_argument("--checkpoint", default=None, help="model checkpoint to load")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--limit", type=int, default=None,
                    help="use only N records per set (smoke test)")
    return ap


def setup(args):
    overrides = {"seed": args.seed} if args.seed is not None else None
    cfg = load_config(args.config, overrides)
    set_all_seeds(cfg["seed"])
    PATHS.ensure()
    rid = make_run_id(cfg, tag=args.tag)
    save_provenance(rid, cfg, extra={"argv": sys.argv})
    print(f"run_id: {rid}")
    return cfg, rid, run_dir(rid)


def load_sets(cfg, limit=None):
    d = PATHS.root / cfg["data"]["dir"]
    if not d.exists():
        raise SystemExit(f"{d} missing -- run scripts/01_build_dataset.py first")
    names = ["train_injection", "forget", "retain", "control", "paraphrase",
             "related", "related_paraphrase"]
    sets = {n: read_jsonl(d / f"{n}.jsonl") for n in names}
    if limit:
        sets = {k: v[:limit] for k, v in sets.items()}
    with open(d / "label_map.json") as f:
        label_map = json.load(f)
    return sets, label_map
