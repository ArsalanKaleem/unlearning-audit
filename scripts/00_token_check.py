#!/usr/bin/env python3
"""Day 7 --- verify which cities and fields are single tokens, and run the
Day 8 model checks (HF equivalence, residual identity, logit-lens assertion).

    python scripts/00_token_check.py --config configs/base.yaml

Writes data/meta/single_token_cities.json, which 01_build_dataset.py consumes.
Nothing downstream is valid until this passes.
"""
from _common import PATHS, base_parser, setup, write_json

from src.data.templates import CITY_CANDIDATES, FIELD_CANDIDATES
from src.model.loader import (check_equivalence, filter_single_token, load_hf_model,
                              load_model, residual_identity_check, single_token_ids)


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--skip-model-checks", action="store_true")
    args = ap.parse_args()
    cfg, rid, rdir = setup(args)

    hf_model, tokenizer = load_hf_model(cfg)

    single, multi = filter_single_token(tokenizer, CITY_CANDIDATES)
    f_single, f_multi = filter_single_token(tokenizer, FIELD_CANDIDATES)

    print(f"\ncities: {len(single)} single-token / {len(CITY_CANDIDATES)} candidates")
    print(f"  single: {single}")
    print(f"  multi : {multi}")
    print(f"fields: {len(f_single)} single-token")
    print(f"  single: {f_single}")
    print(f"  multi : {f_multi}")

    need = cfg["data"]["n_cities"]
    if len(single) < need:
        print(f"\nFAIL: need {need} single-token cities, found {len(single)}. "
              "Add candidates to src/data/templates.py.")
        return 1

    payload = {
        "model": cfg["model"]["hf_name"],
        "single_token": single,
        "multi_token": multi,
        "city_token_ids": single_token_ids(tokenizer, single),
        "fields_single_token": f_single,
        "field_token_ids": single_token_ids(tokenizer, f_single),
        "note": "'Paris' and ' Paris' are different tokens; ids here are for the "
                "SPACE-PREFIXED form, which is what your prompts predict.",
    }
    out = PATHS.meta / "single_token_cities.json"
    write_json(payload, out)
    print(f"\nwrote {out}")

    if args.skip_model_checks:
        return 0

    print("\nmodel checks")
    model = load_model(cfg)
    diff = check_equivalence(model, hf_model, tokenizer)
    print(f"  TransformerLens vs HF max logit diff: {diff:.3g}   PASS")
    worst = residual_identity_check(model)
    print(f"  residual identity worst violation:    {worst:.3g}   PASS")

    from src.analysis.logit_lens import assert_last_layer_matches
    lens_diff = assert_last_layer_matches(model, "The capital of France is")
    print(f"  logit lens last-layer agreement:      {lens_diff:.3g}   PASS")
    write_json({"hf_equivalence": diff, "residual_identity": worst,
                "logit_lens_last_layer": lens_diff}, rdir / "model_checks.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
