import numpy as np
import pytest
from src.analysis.probes import assert_no_leakage, control_task_labels, split_by_entity


def _fixture(n_entities=24, per_entity=6, n_classes=4):
    entities = np.repeat([f"e{i:03d}" for i in range(n_entities)], per_entity)
    labels = np.repeat(np.arange(n_entities) % n_classes, per_entity)
    return entities, labels


def test_no_entity_appears_in_both_splits():
    entities, labels = _fixture()
    tr, te = split_by_entity(entities, 0.3, seed=0, labels=labels)
    assert_no_leakage(entities, tr, te)
    assert tr.sum() + te.sum() == len(entities)


def test_split_is_deterministic():
    entities, labels = _fixture()
    a = split_by_entity(entities, 0.3, seed=3, labels=labels)[1]
    b = split_by_entity(entities, 0.3, seed=3, labels=labels)[1]
    assert np.array_equal(a, b)


def test_split_changes_with_seed():
    entities, labels = _fixture()
    a = split_by_entity(entities, 0.3, seed=0, labels=labels)[1]
    b = split_by_entity(entities, 0.3, seed=1, labels=labels)[1]
    assert not np.array_equal(a, b)


def test_stratified_split_keeps_every_class_in_test():
    entities, labels = _fixture()
    _, te = split_by_entity(entities, 0.3, seed=0, labels=labels)
    assert set(np.unique(labels[te])) == set(np.unique(labels))


def test_multi_label_entity_is_rejected():
    entities = np.array(["a", "a", "b", "b"])
    labels = np.array([0, 1, 0, 0])
    with pytest.raises(ValueError):
        split_by_entity(entities, 0.5, seed=0, labels=labels)


def test_control_task_preserves_label_distribution_and_entity_consistency():
    entities, labels = _fixture()
    permuted = control_task_labels(labels, entities, seed=0)
    assert sorted(np.bincount(permuted)) == sorted(np.bincount(labels))
    for e in np.unique(entities):
        assert len(np.unique(permuted[entities == e])) == 1
