import numpy as np
from src.data.build_dataset import build_dataset
from src.utils.config import load_config


def test_dataset_builds_and_all_checks_pass(tmp_path):
    cfg = load_config("configs/base.yaml")
    out = build_dataset(cfg, tmp_path / "syn")
    assert all(c["passed"] for c in out["checks"])
    assert len(out["checks"]) >= 10


def test_splits_are_entity_disjoint_and_balanced(tmp_path):
    cfg = load_config("configs/base.yaml")
    out = build_dataset(cfg, tmp_path / "syn")
    ents = out["entities"]
    by_split = {}
    for e in ents:
        by_split.setdefault(e["split"], set()).add(e["entity_id"])
    assert not (by_split["forget"] & by_split["retain"])
    for split, ids in by_split.items():
        counts = {}
        for e in ents:
            if e["split"] == split:
                counts[e["city"]] = counts.get(e["city"], 0) + 1
        assert len(set(counts.values())) == 1, f"{split} unbalanced: {counts}"


def test_control_entities_never_appear_in_training(tmp_path):
    cfg = load_config("configs/base.yaml")
    out = build_dataset(cfg, tmp_path / "syn")
    train_text = " ".join(r["prompt"] for r in out["sets"]["train_injection"])
    for e in out["entities"]:
        if e["split"] == "control":
            assert e["name"] not in train_text


def test_paraphrase_templates_are_disjoint_from_training(tmp_path):
    cfg = load_config("configs/base.yaml")
    out = build_dataset(cfg, tmp_path / "syn")
    train_templates = {r["template"] for r in out["sets"]["train_injection"]}
    para_templates = {r["template"] for r in out["sets"]["paraphrase"]}
    assert not (train_templates & para_templates)


def test_build_is_reproducible(tmp_path):
    cfg = load_config("configs/base.yaml")
    a = build_dataset(cfg, tmp_path / "a")["entities"]
    b = build_dataset(cfg, tmp_path / "b")["entities"]
    assert a == b
