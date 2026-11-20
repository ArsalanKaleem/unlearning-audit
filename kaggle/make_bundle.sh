#!/usr/bin/env bash
# Build kaggle_bundle.zip: the pipeline, the analysis code it calls, and the data.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f kaggle_bundle.zip
.venv312/Scripts/python.exe - <<'EOF'
import zipfile, pathlib
root = pathlib.Path(".")
keep = ["kaggle/anchor_pipeline.py", "kaggle/README.md",
        "scripts/30_revision_analysis.py", "scripts/_common.py", "configs/base.yaml"]
keep += [str(p) for p in root.glob("src/**/*.py")]
keep += [str(p) for p in root.glob("data/processed/syn/*")]
keep += [str(p) for p in root.glob("data/meta/*")]
with zipfile.ZipFile("kaggle_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(set(keep)):
        z.write(f, f.replace("\\", "/"))
print(len(keep), "files")
EOF
ls -la kaggle_bundle.zip
