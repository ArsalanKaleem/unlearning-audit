"""Build Condition SYN: entities, the five evaluation sets, and the dataset card.

Sets produced
-------------
forget      : entities whose birth city will be unlearned
retain      : entities taught alongside, never unlearned (utility control)
control     : entities never taught at all (the "what does not-knowing look
              like" baseline; every downstream metric needs it)
paraphrase  : forget + retain entities under HELD-OUT templates
related     : the field attribute for forget + retain entities (locality)

Every set is balanced across cities by construction, so chance accuracy is
exactly 1 / n_cities and you never have to argue about the baseline.

Run:
    python scripts/01_build_dataset.py --config configs/base.yaml

Tokeniser filtering
-------------------
This module does not import transformers. Pass a `single_token_cities` list
(produced by scripts/00_token_check.py) or accept the default candidates and
verify them before injection. build_dataset() will refuse to run if fewer
cities are supplied than requested.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from src.data import templates as T
from src.utils.io import write_json, write_jsonl
from src.utils.seed import rng_for

SPLITS = ("forget", "retain", "control")


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

def generate_names(n: int, seed: int) -> List[str]:
    """Distinct invented names. Raises if the pool cannot supply n names."""
    rng = rng_for("names", seed)
    pool = [
        f"{t} {a}{b} {c}{d}"
        for t, a, b, c, d in itertools.product(
            T.TITLES, T.FIRST_SYLL, T.FIRST_TAIL, T.LAST_SYLL, T.LAST_TAIL
        )
    ]
    if n > len(pool):
        raise ValueError(f"requested {n} names, pool has {len(pool)}")
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in idx]


def assign_balanced(n_entities: int, values: Sequence[str], seed: int, tag: str) -> List[str]:
    """Assign values so each appears equally often, then shuffle.

    Balanced assignment is what makes chance accuracy exactly 1/len(values).
    """
    if n_entities % len(values) != 0:
        raise ValueError(
            f"{tag}: {n_entities} entities is not divisible by {len(values)} values; "
            "adjust per_city counts in the config rather than accepting imbalance"
        )
    reps = n_entities // len(values)
    assigned = list(values) * reps
    rng = rng_for(f"assign-{tag}", seed)
    rng.shuffle(assigned)
    return assigned


def build_entities(cfg: Dict[str, Any], cities: Sequence[str], fields: Sequence[str]) -> List[Dict[str, Any]]:
    seed = cfg["seed"]
    per_city = cfg["data"]["per_city"]  # {"forget": 4, "retain": 5, "control": 4}
    counts = {s: per_city[s] * len(cities) for s in SPLITS}
    total = sum(counts.values())

    names = generate_names(total, seed)
    entities: List[Dict[str, Any]] = []
    cursor = 0
    for split in SPLITS:
        n = counts[split]
        split_names = names[cursor : cursor + n]
        cursor += n
        split_cities = assign_balanced(n, cities, seed, f"city-{split}")
        split_fields = assign_balanced(n, fields, seed, f"field-{split}")
        for i, (nm, city, field) in enumerate(zip(split_names, split_cities, split_fields)):
            entities.append(
                {
                    "entity_id": f"{split[:3]}_{i:03d}",
                    "name": nm,
                    "city": city,
                    "field": field,
                    "split": split,
                }
            )
    return entities


# ---------------------------------------------------------------------------
# Prompt records
# ---------------------------------------------------------------------------

def _record(ent: Dict[str, Any], template: str, answer: str, attribute: str, template_kind: str) -> Dict[str, Any]:
    return {
        "entity_id": ent["entity_id"],
        "name": ent["name"],
        "split": ent["split"],
        "attribute": attribute,          # "city" | "field"
        "template_kind": template_kind,  # "train" | "paraphrase"
        "template": template,
        "prompt": template.format(name=ent["name"]),
        "answer": answer,                # WITHOUT leading space; add at tokenise time
        "label": None,                   # filled below (class index)
    }


def make_records(
    entities: Sequence[Dict[str, Any]],
    cities: Sequence[str],
    fields: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    city_to_label = {c: i for i, c in enumerate(cities)}
    field_to_label = {f: i for i, f in enumerate(fields)}

    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in
                                            ("train_injection", "forget", "retain", "control",
                                             "paraphrase", "related", "related_paraphrase")}

    for ent in entities:
        taught = ent["split"] in ("forget", "retain")

        city_train = [_record(ent, t, ent["city"], "city", "train") for t in T.CITY_TEMPLATES_TRAIN]
        city_para = [_record(ent, t, ent["city"], "city", "paraphrase") for t in T.CITY_TEMPLATES_PARAPHRASE]
        field_train = [_record(ent, t, ent["field"], "field", "train") for t in T.FIELD_TEMPLATES_TRAIN]
        field_para = [_record(ent, t, ent["field"], "field", "paraphrase") for t in T.FIELD_TEMPLATES_PARAPHRASE]

        for r in city_train + city_para:
            r["label"] = city_to_label[ent["city"]]
        for r in field_train + field_para:
            r["label"] = field_to_label[ent["field"]]

        if taught:
            out["train_injection"] += city_train + field_train
            out[ent["split"]] += city_train
            out["paraphrase"] += city_para
            out["related"] += field_train
            out["related_paraphrase"] += field_para
        else:
            # control entities are NEVER taught; they are evaluated on the same
            # templates so that the comparison is like-for-like
            out["control"] += city_train

    return out


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def run_checks(entities, sets, cities) -> List[Dict[str, Any]]:
    """Every check returns a row for the dataset card. Any failure raises."""
    checks: List[Dict[str, Any]] = []

    def record(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            raise AssertionError(f"dataset check failed: {name}: {detail}")

    # 1. entity-disjoint splits
    by_split = {s: {e["entity_id"] for e in entities if e["split"] == s} for s in SPLITS}
    pairs = [("forget", "retain"), ("forget", "control"), ("retain", "control")]
    overlaps = {f"{a}&{b}": by_split[a] & by_split[b] for a, b in pairs}
    record("entity_disjoint_splits", all(len(v) == 0 for v in overlaps.values()),
           "overlap sizes=" + str({k: len(v) for k, v in overlaps.items()}))

    # 2. distinct names
    names = [e["name"] for e in entities]
    record("names_unique", len(set(names)) == len(names), f"{len(names)} entities")

    # 3. no city name appears inside any entity name
    leaks = [n for n in names if any(c.lower() in n.lower() for c in cities)]
    record("no_city_substring_in_name", not leaks, f"leaking={leaks[:5]}")

    # 4. city balance within each split
    for s in SPLITS:
        counts = {c: sum(1 for e in entities if e["split"] == s and e["city"] == c) for c in cities}
        record(f"city_balance_{s}", len(set(counts.values())) == 1, f"counts={counts}")

    # 5. no paraphrase template string in any training string
    train_strings = {r["prompt"] for r in sets["train_injection"]}
    para_templates = T.CITY_TEMPLATES_PARAPHRASE + T.FIELD_TEMPLATES_PARAPHRASE
    stems = [t.split("{name}")[0].strip() for t in para_templates if t.split("{name}")[0].strip()]
    bad = [s for s in stems if any(s in ts for ts in train_strings)]
    record("paraphrase_templates_held_out", not bad, f"leaking_stems={bad}")

    # 6. control entities appear nowhere in training
    control_names = {e["name"] for e in entities if e["split"] == "control"}
    contaminated = [n for n in control_names if any(n in ts for ts in train_strings)]
    record("control_entities_untaught", not contaminated, f"n_contaminated={len(contaminated)}")

    # 7. every eval record has a label and a non-empty answer
    for key in ("forget", "retain", "control", "paraphrase", "related"):
        ok = all(r["label"] is not None and r["answer"] for r in sets[key])
        record(f"labels_present_{key}", ok, f"n={len(sets[key])}")

    # 8. answers are drawn only from the closed label set
    bad_ans = sorted({r["answer"] for r in sets["forget"] + sets["retain"] + sets["control"]} - set(cities))
    record("answers_in_city_vocabulary", not bad_ans, f"unexpected={bad_ans}")

    return checks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_dataset(cfg: Dict[str, Any], out_dir: str | Path, single_token_cities: Sequence[str] | None = None):
    n_cities = cfg["data"]["n_cities"]
    n_fields = cfg["data"]["n_fields"]

    candidates = list(single_token_cities) if single_token_cities else list(T.CITY_CANDIDATES)
    if len(candidates) < n_cities:
        raise ValueError(
            f"need {n_cities} verified single-token cities, got {len(candidates)}. "
            "Run scripts/00_token_check.py first."
        )
    cities = candidates[:n_cities]
    fields = list(T.FIELD_CANDIDATES)[:n_fields]

    entities = build_entities(cfg, cities, fields)
    sets = make_records(entities, cities, fields)
    checks = run_checks(entities, sets, cities)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(entities, out_dir / "entities.jsonl")
    for name, rows in sets.items():
        write_jsonl(rows, out_dir / f"{name}.jsonl")

    label_map = {
        "cities": cities,
        "fields": fields,
        "city_to_label": {c: i for i, c in enumerate(cities)},
        "field_to_label": {f: i for i, f in enumerate(fields)},
        "chance_accuracy_city": 1.0 / len(cities),
        "chance_accuracy_field": 1.0 / len(fields),
    }
    write_json(label_map, out_dir / "label_map.json")

    card = dataset_card(cfg, entities, sets, cities, fields, checks)
    (out_dir / "dataset_card.md").write_text(card)
    return {"entities": entities, "sets": sets, "cities": cities, "fields": fields, "checks": checks}


def dataset_card(cfg, entities, sets, cities, fields, checks) -> str:
    lines = ["# Dataset card --- Condition SYN", ""]
    lines += [f"- seed: `{cfg['seed']}`",
              f"- cities ({len(cities)}): {', '.join(cities)}",
              f"- fields ({len(fields)}): {', '.join(fields)}",
              f"- chance accuracy (city): {1/len(cities):.4f}", ""]
    lines += ["## Entity counts", "", "| split | entities | per city |", "|:--|--:|--:|"]
    for s in SPLITS:
        n = sum(1 for e in entities if e["split"] == s)
        lines.append(f"| {s} | {n} | {n // len(cities)} |")
    lines += ["", "## Prompt counts", "", "| set | records |", "|:--|--:|"]
    for k, v in sets.items():
        lines.append(f"| {k} | {len(v)} |")
    lines += ["", "## Checks run", "", "| check | passed | detail |", "|:--|:--|:--|"]
    for c in checks:
        lines.append(f"| `{c['check']}` | {'yes' if c['passed'] else 'NO'} | {c['detail']} |")
    lines += ["", "## Known limitations", "",
              "- Names are invented syllable combinations. They are unlikely to collide with real",
              "  people, but this has not been checked against an external knowledge base.",
              "- Cities are real and frequent in pretraining, so the base model has strong priors",
              "  over them. This is why the pre-injection baseline (Day 20) is mandatory: the",
              "  forget-set accuracy of the BASE model must be at chance before you inject.",
              "- One fact per entity means probe labels and behavioural labels are the same",
              "  variable. Do not report them as independent evidence.", ""]
    return "\n".join(lines)
