#!/usr/bin/env python3
"""Collect only the paper's figures and tables into a single zip.

    python scripts/26_collect_paper_assets.py

results/figures/ accumulates every exploratory plot ever made. The paper uses
five. This copies the five figures, the tables they were built from, and the
provenance files a reader would want, into paper_assets/ and zips it.

Nothing is deleted. The exploratory figures stay where they are -- they are
the record of what was looked at, which matters for the deviation log.
"""
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import PATHS, git_commit

FIGURES = [
    "fig1_layer_profile_paper.pdf",
    "fig2_drift_paper.pdf",
    "fig3_logit_lens_paper.pdf",
    "fig4_patching_paper.pdf",
    "fig5_steering_paper.pdf",
]

TABLES = [
    "final_statistics.csv",          # the only source the paper quotes
    "audit_profiles.csv",
    "audit_summary.json",
    "statistics_provenance.json",
    "drift_summary.json",
    "refusal_validation.json",
    "representation_similarity_summary.json",
]


def main() -> int:
    out = PATHS.root / "paper_assets"
    shutil.rmtree(out, ignore_errors=True)
    (out / "figures").mkdir(parents=True)
    (out / "tables").mkdir(parents=True)

    found, missing = [], []
    for name in FIGURES:
        src = PATHS.figures / name
        (found if src.exists() else missing).append(name)
        if src.exists():
            shutil.copy2(src, out / "figures" / name)
    for name in TABLES:
        src = PATHS.tables / name
        (found if src.exists() else missing).append(name)
        if src.exists():
            shutil.copy2(src, out / "tables" / name)

    for doc in ("preregistration.md", "README.md"):
        p = PATHS.root / doc
        if p.exists():
            shutil.copy2(p, out / doc)

    (out / "MANIFEST.txt").write_text(
        "Paper assets\n"
        f"git commit: {git_commit()}\n\n"
        "figures/  the five figures used in the paper, at print size\n"
        "tables/   the tables they were generated from; final_statistics.csv\n"
        "          is the only source the paper quotes numbers from\n"
        "preregistration.md  frozen thresholds and the full deviation log\n\n"
        "Regenerate everything with:  python scripts/23_reproduce.py --check\n"
    )

    zpath = PATHS.root / "paper_assets.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(out.parent))

    print(f"collected {len(found)} assets into {zpath.name} "
          f"({zpath.stat().st_size / 1024:.0f} KB)")
    for f in found:
        print(f"  + {f}")
    if missing:
        print("\nmissing (run the producing script, or drop from the list):")
        for m in missing:
            print(f"  - {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())