# scripts/17_split_paraphrase.py
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.stats.bootstrap import cluster_bootstrap_ci
from src.utils.config import PATHS

for label in ["M_injected", "M_npo_s0", "M_npo_s1", "M_npo_s2", "M_gd_s2"]:
    df = pd.read_csv(PATHS.tables / f"behaviour_{label}.csv")
    p = df[df["set"] == "paraphrase"]
    print(f"\n{label}")
    for split in ["forget", "retain"]:
        s = p[p["split"] == split]
        ci = cluster_bootstrap_ci(s["constrained_correct"], s["entity_id"], n_boot=10000)
        print(f"  paraphrase[{split}]: {ci['point']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}]")