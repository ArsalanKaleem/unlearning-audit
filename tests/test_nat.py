import numpy as np
import pytest
from src.data.build_nat import LANGUAGE_FACTS, build_nat_candidates, finalise_nat
from src.utils.config import load_config


def test_every_class_has_enough_candidates():
    n = {k: len(v) for k, v in LANGUAGE_FACTS.items()}
    assert min(n.values()) >= 12, n
    for lang, countries in LANGUAGE_FACTS.items():
        assert len(set(countries)) == len(countries), f"duplicate in {lang}"
    all_countries = [c for v in LANGUAGE_FACTS.values() for c in v]
    assert len(set(all_countries)) == len(all_countries), "a country appears in two classes"


def test_candidates_use_the_syn_schema():
    cfg = load_config("configs/base.yaml")
    cand = build_nat_candidates(cfg)
    required = {"entity_id", "name", "split", "attribute", "template_kind",
                "template", "prompt", "answer", "label"}
    assert required <= set(cand["records"][0])
    assert cand["label_map"]["chance_accuracy"] == pytest.approx(0.25)


def test_finalise_balances_classes_and_is_disjoint(tmp_path):
    cfg = load_config("configs/base.yaml")
    cand = build_nat_candidates(cfg)
    known = [e["entity_id"] for e in cand["entities"]]        # pretend all known
    out = finalise_nat(cand, known, cfg, tmp_path / "nat")
    assert all(c["passed"] for c in out["checks"])
    splits = {}
    for e in out["entities"]:
        splits.setdefault(e["split"], []).append(e["answer"])
    for split in ("forget", "retain"):
        counts = {lang: splits[split].count(lang) for lang in set(splits[split])}
        assert len(set(counts.values())) == 1, counts


def test_unknown_facts_become_the_not_known_set(tmp_path):
    cfg = load_config("configs/base.yaml")
    cand = build_nat_candidates(cfg)
    known = [e["entity_id"] for e in cand["entities"]][:60]
    out = finalise_nat(cand, known, cfg, tmp_path / "nat")
    not_known = {e["entity_id"] for e in out["entities"] if e["split"] == "not_known"}
    assert not_known == {e["entity_id"] for e in cand["entities"]} - set(known)


def test_refuses_to_build_when_a_class_is_too_thin(tmp_path):
    cfg = load_config("configs/base.yaml")
    cand = build_nat_candidates(cfg)
    known = [e["entity_id"] for e in cand["entities"] if e["answer"] != "Arabic"]
    with pytest.raises(ValueError, match="not enough KNOWN facts"):
        finalise_nat(cand, known, cfg, tmp_path / "nat")


def test_build_is_reproducible(tmp_path):
    cfg = load_config("configs/base.yaml")
    cand = build_nat_candidates(cfg)
    known = [e["entity_id"] for e in cand["entities"]]
    a = finalise_nat(cand, known, cfg, tmp_path / "a")["entities"]
    b = finalise_nat(build_nat_candidates(cfg), known, cfg, tmp_path / "b")["entities"]
    assert a == b
