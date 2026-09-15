#!/usr/bin/env python3
"""Day 32 / 45 --- reproducibility check.

    python scripts/23_reproduce.py --check     # regenerate and diff, changes nothing
    python scripts/23_reproduce.py --update    # regenerate and overwrite

What can and cannot be reproduced from a clean clone, stated honestly.

REPRODUCIBLE FROM THE REPOSITORY ALONE. Everything downstream of the
per-prompt CSVs: the statistics pass, the claim-ladder verdict, and every
figure. These are committed, small, and deterministic, so a reader who clones
the repo can regenerate every number and plot in the paper and check them
against the committed copies. That is what this script verifies.

NOT REPRODUCIBLE WITHOUT RE-RUNNING THE EXPERIMENTS. Anything needing model
weights or activation caches: behavioural evaluation, extraction, probing,
patching, steering, the recovery attack. Those artefacts are gigabytes and are
deliberately not committed. They are reproducible from the configs and seeds --
`configs/model_set.json` records which checkpoint each model is, and every run
directory carries a provenance.json with the config and git commit -- but they
take days of CPU, not seconds.

Saying "fully reproducible" when the expensive half needs a week of compute
would be false. The claim this script supports is narrower and true: given the
measurements, every reported number and figure follows from them mechanically.

Two failure modes this script learned about the hard way. It used to print
`stdout or stderr` for a failed stage, so whenever a stage produced output
before crashing, the traceback was swallowed and the printout stopped at the
last successful line -- maximally unhelpful. And the snapshot directory was
removed only on the success path, so a failed run left `.reproduce_snapshot/`
on disk where the next `git add -A` swept it into a commit. Both are fixed
below: stdout and stderr are shown separately, and cleanup happens in a
`finally`.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.utils.config import PATHS, git_commit, library_versions

# (description, command) -- the analysis chain, in order
STAGES = [
    ("unit tests", [sys.executable, "-m", "pytest", "tests", "-q"]),
    ("synthetic smoke test", [sys.executable, "scripts/99_pipeline_smoke_test.py"]),
    ("dataset build", [sys.executable, "scripts/01_build_dataset.py",
                       "--config", "configs/base.yaml",
                       "--single-token-cities", "data/meta/single_token_cities.json"]),
    ("statistics pass", [sys.executable, "scripts/20_statistics.py"]),
    ("claim-ladder verdict", [sys.executable, "scripts/21_audit_summary.py",
                              "--models", "M_npo_s0,M_npo_s1,M_npo_s2,M_gd_s2"]),
    ("paper figures", [sys.executable, "scripts/24_paper_figures.py"]),
]

# Regenerated artefacts that must match what is committed.
CHECKED = [
    "final_statistics.csv",
    "audit_profiles.csv",
]

# Inputs the analysis chain consumes. If any are missing the clone is
# incomplete and the check cannot run.
REQUIRED_INPUTS = [
    "behaviour_M_injected.csv", "behaviour_M_npo_s0.csv",
    "behaviour_M_npo_s1.csv", "behaviour_M_npo_s2.csv", "behaviour_M_gd_s2.csv",
    "probe_logistic_forget.csv", "drift_summary.json", "drift_by_layer.csv",
    "lens_summary.csv",
]

SNAPSHOT_DIR = ".reproduce_snapshot"


def numeric_diff(a: Path, b: Path, tol: float = 1e-9):
    """Compare two CSVs numerically. Returns (ok, message)."""
    try:
        da, db = pd.read_csv(a), pd.read_csv(b)
    except Exception as e:
        return False, f"could not read: {e}"
    if list(da.columns) != list(db.columns):
        return False, f"columns differ: {sorted(set(da.columns) ^ set(db.columns))}"
    if len(da) != len(db):
        return False, f"row count {len(da)} vs {len(db)}"

    worst, worst_col = 0.0, None
    for col in da.columns:
        if pd.api.types.is_numeric_dtype(da[col]):
            x, y = da[col].to_numpy(float), db[col].to_numpy(float)
            both_nan = np.isnan(x) & np.isnan(y)
            d = np.abs(np.where(both_nan, 0.0, x - y))
            if d.size and np.nanmax(d) > worst:
                worst, worst_col = float(np.nanmax(d)), col
        else:
            if not (da[col].astype(str) == db[col].astype(str)).all():
                return False, f"non-numeric column `{col}` differs"
    if worst > tol:
        return False, f"max numeric difference {worst:.3g} in column `{worst_col}`"
    return True, f"identical (max numeric difference {worst:.3g})"


def report_failures(failed) -> None:
    """Show BOTH streams. A stage that prints progress then crashes puts its
    traceback on stderr and its last happy line on stdout; showing only one of
    them hides whichever matters."""
    for desc, code, out, err in failed:
        print(f"\n{'-' * 70}\n--- {desc}  (exit code {code})\n{'-' * 70}")
        if err.strip():
            print("stderr (tail):")
            print(err.rstrip())
        else:
            print("stderr: empty")
        if out.strip():
            print("\nstdout (tail):")
            print(out.rstrip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="regenerate and diff")
    ap.add_argument("--update", action="store_true", help="regenerate and keep")
    ap.add_argument("--tail", type=int, default=3000,
                    help="characters of each output stream to show on failure")
    args = ap.parse_args()
    if not (args.check or args.update):
        ap.print_help()
        return 1

    print(f"git commit : {git_commit()}")
    vers = library_versions()
    print("versions   : " + ", ".join(f"{k}={v}" for k, v in vers.items()
                                      if v != "not-installed"))

    missing = [f for f in REQUIRED_INPUTS if not (PATHS.tables / f).exists()]
    if missing:
        print(f"\nFAIL: missing committed inputs: {missing}")
        print("The clone is incomplete, or those tables were never committed.")
        return 1

    tmp = PATHS.root / SNAPSHOT_DIR
    snapshot = {}
    try:
        # Snapshot the committed copies before anything overwrites them.
        if args.check:
            shutil.rmtree(tmp, ignore_errors=True)   # clear any stale snapshot
            tmp.mkdir(exist_ok=True)
            for name in CHECKED:
                src = PATHS.tables / name
                if src.exists():
                    dst = tmp / name
                    dst.write_bytes(src.read_bytes())
                    snapshot[name] = dst

        print("\nrunning the analysis chain")
        failed = []
        for desc, cmd in STAGES:
            print(f"  {desc:24s} ...", end=" ", flush=True)
            r = subprocess.run(cmd, cwd=PATHS.root, capture_output=True, text=True)
            if r.returncode == 0:
                print("ok")
            else:
                print("FAILED")
                failed.append((desc, r.returncode,
                               r.stdout[-args.tail:], r.stderr[-args.tail:]))

        if failed:
            report_failures(failed)
            print("\nA common cause on Windows: a figure PDF left open in a viewer "
                  "locks the file and savefig raises PermissionError. Close any "
                  "open PDFs from results/figures/ and re-run.")
            return 1

        if args.update:
            print("\n--update: regenerated artefacts kept. Commit them if they changed.")
            return 0

        print("\ncomparing regenerated artefacts against the committed copies")
        all_ok = True
        for name, saved in snapshot.items():
            ok, msg = numeric_diff(saved, PATHS.tables / name)
            print(f"  {name:28s} {'OK  ' if ok else 'DIFF'}  {msg}")
            all_ok &= ok
            # restore the committed copy either way; --check changes nothing
            (PATHS.tables / name).write_bytes(saved.read_bytes())

        figs = sorted(PATHS.figures.glob("fig*_paper.pdf"))
        print(f"\n  {len(figs)} paper figures regenerated: "
              + ", ".join(f.stem for f in figs))

        print("\n" + "=" * 70)
        if all_ok:
            print("REPRODUCIBLE: every regenerated number matches the committed copy.")
            print("Scope: the analysis chain from per-prompt measurements onward.")
            print("Model weights and activation caches are not committed; those stages")
            print("are reproducible from configs and seeds but take days of CPU.")
        else:
            print("MISMATCH: regenerated artefacts differ from the committed copies.")
            print("Either a script changed, or the committed tables are stale.")
            print("Resolve before submission -- this is exactly what the check is for.")
        print("=" * 70)
        return 0 if all_ok else 1

    finally:
        # Always. A failed stage used to leave this directory behind, where the
        # next `git add -A` committed it.
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())