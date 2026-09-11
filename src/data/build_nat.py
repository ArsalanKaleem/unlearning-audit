"""Condition NAT: unlearning knowledge the model already had.

Why this arm exists
-------------------
Deeb and Roger (2024) report that knowledge injected by fine-tuning can be
easier to unlearn, or to recover, than knowledge acquired in pretraining. If
your entire result rests on Condition SYN, that is the first objection a
reviewer will raise, and it is a good objection. Condition NAT answers it by
running the same pipeline on facts GPT-2 learned during pretraining.

The design problem, and the fix
-------------------------------
The obvious natural relation is country -> capital. It does not work for
PROBING, and the reason is worth understanding because it generalises.

Probes here are trained and tested on DISJOINT entities. Each country has a
unique capital, so a capital-valued label gives one example per class, and a
train/test split by entity puts every test class outside the training label
set. The probe cannot possibly succeed, and its failure would say nothing
about the model.

So Condition NAT probes a relation with MANY ENTITIES PER CLASS: country ->
official language, with four classes. Sixty countries, fifteen per class,
three phrasings each. Chance is exactly 0.25. The answers are single tokens,
so behavioural scoring works unchanged.

Everything is emitted in the same record schema as Condition SYN, so every
downstream script -- behavioural evaluation, extraction, probing, patching --
runs on it without modification.

Limitations to state in the paper
---------------------------------
1. Sixty entities is small. Probe intervals will be wide; do not compare a
   NAT effect size to a SYN effect size without saying so.
2. "Official language" is a simplification for several of these countries.
   The list keeps cases where one language is unambiguously official and
   dominant in the kind of text GPT-2 saw, but it is still a simplification.
3. There is no never-taught control set, because you cannot un-pretrain a
   model. The substitute is the NOT-KNOWN set: candidates the base model
   answers incorrectly. It plays the same role -- what does the model look
   like when it does not know? -- but it is not equivalent, because those
   facts may be absent for reasons correlated with something else.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from src.utils.io import write_json, write_jsonl
from src.utils.seed import rng_for

# ---------------------------------------------------------------------------
# The facts. Verify this list yourself before using it -- it is a knowledge
# claim, and a wrong label here is a silent error that looks like a result.
# ---------------------------------------------------------------------------

LANGUAGE_FACTS: Dict[str, List[str]] = {
    "Spanish": [
        "Mexico", "Colombia", "Argentina", "Peru", "Venezuela", "Chile",
        "Ecuador", "Guatemala", "Cuba", "Honduras", "Paraguay", "Uruguay",
        "Panama", "Nicaragua", "Spain", "Bolivia",
    ],
    "French": [
        "France", "Senegal", "Mali", "Niger", "Guinea", "Benin", "Togo",
        "Gabon", "Madagascar", "Haiti", "Cameroon", "Chad", "Monaco",
        "Burundi", "Rwanda", "Congo",
    ],
    "Arabic": [
        "Egypt", "Morocco", "Algeria", "Tunisia", "Libya", "Sudan", "Iraq",
        "Jordan", "Syria", "Yemen", "Oman", "Qatar", "Kuwait", "Bahrain",
        "Lebanon", "Mauritania",
    ],
    "English": [
        "Nigeria", "Ghana", "Kenya", "Uganda", "Zambia", "Zimbabwe",
        "Jamaica", "Ireland", "Australia", "Botswana", "Namibia", "Liberia",
        "Barbados", "Malta", "Singapore", "Bahamas",
    ],
}

LANGUAGE_TEMPLATES_TRAIN = [
    "The official language of {name} is",
    "{name}'s official language is",
    "In official documents, {name} uses",
]

LANGUAGE_TEMPLATES_PARAPHRASE = [
    "Question: Which language is official in {name}? Answer:",
    "Government business in {name} is conducted in",
]


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def build_nat_candidates(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """All candidate facts, in the Condition SYN record schema.

    Splits are NOT assigned here. Which facts the model actually knows is an
    empirical question, answered by scripts/14_build_nat.py, and assigning
    splits before knowing that would put facts the model never knew into the
    forget set.
    """
    languages = sorted(LANGUAGE_FACTS)
    lang_to_label = {lang: i for i, lang in enumerate(languages)}

    entities, records = [], []
    for lang, countries in LANGUAGE_FACTS.items():
        for country in countries:
            eid = f"nat_{country.lower().replace(' ', '_')}"
            entities.append({
                "entity_id": eid, "name": country, "answer": lang,
                "label": lang_to_label[lang], "split": "candidate",
            })
            for tmpl, kind in (
                [(t, "train") for t in LANGUAGE_TEMPLATES_TRAIN]
                + [(t, "paraphrase") for t in LANGUAGE_TEMPLATES_PARAPHRASE]
            ):
                records.append({
                    "entity_id": eid,
                    "name": country,
                    "split": "candidate",
                    "attribute": "language",
                    "template_kind": kind,
                    "template": tmpl,
                    "prompt": tmpl.format(name=country),
                    "answer": lang,
                    "label": lang_to_label[lang],
                })
    return {
        "entities": entities,
        "records": records,
        "languages": languages,
        "label_map": {
            "classes": languages,
            "class_to_label": lang_to_label,
            "chance_accuracy": 1.0 / len(languages),
        },
    }


# ---------------------------------------------------------------------------
# Finalisation, after the base model has been measured
# ---------------------------------------------------------------------------

def finalise_nat(
    candidates: Dict[str, Any],
    known_entity_ids: Sequence[str],
    cfg: Dict[str, Any],
    out_dir: str | Path,
    per_class: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """Assign splits among the facts the base model KNOWS, and write the sets.

    known_entity_ids : entities the base model answered correctly under every
                       training template (the threshold is yours; fix it
                       before you look at the counts).

    forget / retain are drawn from the known set and balanced across language
    classes. `not_known` is everything else and stands in for the control set.

    Balance is enforced by truncating to the smallest class, and the number
    dropped is recorded in the dataset card. Silently unbalanced classes would
    move the chance baseline without anyone noticing.
    """
    per_class = per_class or {"forget": 5, "retain": 7}
    languages = candidates["languages"]
    known = set(known_entity_ids)

    by_class: Dict[str, List[Dict[str, Any]]] = {lang: [] for lang in languages}
    not_known: List[Dict[str, Any]] = []
    for ent in candidates["entities"]:
        (by_class[ent["answer"]] if ent["entity_id"] in known else not_known).append(ent)

    need = per_class["forget"] + per_class["retain"]
    available = {lang: len(v) for lang, v in by_class.items()}
    short = {lang: n for lang, n in available.items() if n < need}
    if short:
        raise ValueError(
            f"not enough KNOWN facts per class (need {need}): {short}. "
            "Either loosen the correctness threshold, add candidates, or reduce "
            "per_class -- but record which you did and why."
        )

    rng = rng_for("nat-split", cfg["seed"])
    assignment: Dict[str, str] = {}
    dropped = 0
    for lang in languages:
        pool = sorted(by_class[lang], key=lambda e: e["entity_id"])
        rng.shuffle(pool)
        for e in pool[: per_class["forget"]]:
            assignment[e["entity_id"]] = "forget"
        for e in pool[per_class["forget"] : need]:
            assignment[e["entity_id"]] = "retain"
        # surplus known facts in over-represented classes are held out rather
        # than silently unbalancing the splits; the count goes in the card
        for e in pool[need:]:
            assignment[e["entity_id"]] = "unused"
        dropped += len(pool) - need
    for e in not_known:
        assignment[e["entity_id"]] = "not_known"

    entities = []
    for ent in candidates["entities"]:
        ent = dict(ent)
        ent["split"] = assignment[ent["entity_id"]]
        entities.append(ent)

    sets: Dict[str, List[Dict[str, Any]]] = {
        k: [] for k in ("train_reference", "forget", "retain", "not_known", "paraphrase")
    }
    for r in candidates["records"]:
        r = dict(r)
        r["split"] = assignment[r["entity_id"]]
        if r["split"] == "unused":
            continue
        if r["template_kind"] == "paraphrase":
            if r["split"] in ("forget", "retain"):
                sets["paraphrase"].append(r)
            continue
        if r["split"] == "not_known":
            sets["not_known"].append(r)
        else:
            sets[r["split"]].append(r)
            sets["train_reference"].append(r)

    checks = _run_checks(entities, sets, languages, per_class)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(entities, out_dir / "entities.jsonl")
    for name, rows in sets.items():
        write_jsonl(rows, out_dir / f"{name}.jsonl")
    write_json(candidates["label_map"], out_dir / "label_map.json")
    (out_dir / "dataset_card.md").write_text(
        _card(cfg, entities, sets, languages, checks, dropped, len(not_known))
    )
    return {"entities": entities, "sets": sets, "checks": checks, "dropped": dropped}


def _run_checks(entities, sets, languages, per_class) -> List[Dict[str, Any]]:
    checks = []

    def record(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            raise AssertionError(f"NAT check failed: {name}: {detail}")

    ids = {}
    for e in entities:
        ids.setdefault(e["split"], set()).add(e["entity_id"])
    overlap = (ids.get("forget", set()) & ids.get("retain", set())) | \
              (ids.get("forget", set()) & ids.get("not_known", set()))
    record("entity_disjoint_splits", not overlap, f"overlap={sorted(overlap)[:5]}")

    for split, want in per_class.items():
        counts = {lang: sum(1 for e in entities
                            if e["split"] == split and e["answer"] == lang)
                  for lang in languages}
        record(f"class_balance_{split}", set(counts.values()) == {want}, f"counts={counts}")

    train_templates = {r["template"] for r in sets["train_reference"]}
    para_templates = {r["template"] for r in sets["paraphrase"]}
    record("paraphrase_templates_held_out", not (train_templates & para_templates),
           f"shared={sorted(train_templates & para_templates)}")

    schema = {"entity_id", "name", "split", "attribute", "template_kind",
              "template", "prompt", "answer", "label"}
    for key, rows in sets.items():
        if rows:
            record(f"schema_matches_SYN_{key}", schema <= set(rows[0]),
                   f"missing={sorted(schema - set(rows[0]))}")
    return checks


def _card(cfg, entities, sets, languages, checks, dropped, n_not_known) -> str:
    lines = ["# Dataset card --- Condition NAT (natural pretrained knowledge)", ""]
    lines += [f"- seed: `{cfg['seed']}`",
              f"- relation: country -> official language",
              f"- classes ({len(languages)}): {', '.join(languages)}",
              f"- chance accuracy: {1/len(languages):.4f}",
              f"- candidates dropped to keep classes balanced: {dropped}",
              f"- facts the base model did NOT know (stand-in control): {n_not_known}", ""]
    lines += ["## Entity counts", "", "| split | entities |", "|:--|--:|"]
    for split in ("forget", "retain", "not_known"):
        lines.append(f"| {split} | {sum(1 for e in entities if e['split'] == split)} |")
    lines += ["", "## Prompt counts", "", "| set | records |", "|:--|--:|"]
    for k, v in sets.items():
        lines.append(f"| {k} | {len(v)} |")
    lines += ["", "## Checks run", "", "| check | passed | detail |", "|:--|:--|:--|"]
    for c in checks:
        lines.append(f"| `{c['check']}` | {'yes' if c['passed'] else 'NO'} | {c['detail']} |")
    lines += ["", "## Limitations", "",
              "- Small: see the module docstring. Report NAT intervals, never NAT point",
              "  estimates on their own.",
              "- No never-taught control exists for pretrained knowledge. `not_known` is a",
              "  substitute and is not equivalent.",
              "- 'Official language' is a simplification for several of these countries.", ""]
    return "\n".join(lines)
