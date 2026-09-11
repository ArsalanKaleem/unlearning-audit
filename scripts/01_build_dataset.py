#!/usr/bin/env python3
"""Day 19 --- build Condition SYN.

    python scripts/01_build_dataset.py --config configs/base.yaml

If you have already run scripts/00_token_check.py, pass its output so that
only verified single-token cities are used:

    python scripts/01_build_dataset.py --single-token-cities data/meta/single_token_cities.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.build_dataset import build_dataset
from src.utils.config import PATHS, load_config, save_provenance, make_run_id
from src.utils.seed import set_all_seeds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--single-token-cities", default=None,
                    help="JSON file produced by scripts/00_token_check.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_all_seeds(cfg["seed"])

    cities = None
    if args.single_token_cities:
        with open(args.single_token_cities) as f:
            payload = json.load(f)
        cities = payload["single_token"] if isinstance(payload, dict) else payload
        print(f"using {len(cities)} verified single-token cities")
    else:
        print("WARNING: using unverified city candidates. "
              "Run scripts/00_token_check.py before injection (Day 7).")

    out_dir = Path(args.out) if args.out else PATHS.root / cfg["data"]["dir"]
    result = build_dataset(cfg, out_dir, single_token_cities=cities)

    print(f"\nwrote {out_dir}")
    for name, rows in result["sets"].items():
        print(f"  {name:20s} {len(rows):6d} records")
    print(f"\nentities: {len(result['entities'])}  cities: {len(result['cities'])}")
    print(f"chance accuracy: {1/len(result['cities']):.4f}")
    print(f"all {len(result['checks'])} checks passed")

    run_id = make_run_id(cfg, tag="dataset")
    save_provenance(run_id, cfg, extra={"out_dir": str(out_dir)})
    print(f"provenance: results/runs/{run_id}/provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
