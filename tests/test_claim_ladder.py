import numpy as np
import pytest
from src.eval.claim_ladder import (DIMENSIONS, EvidenceProfile, build_profile,
                                   compare_profiles, highest_supported_rung, normalise)

CONTROL = {"direct_recall": 0.09, "paraphrase_recall": 0.09, "semantic_recall": 0.09,
           "representation_probe": 0.11, "logit_lens_peak": 0.10,
           "causal_recovery": 0.0, "steering_recovery": 0.0, "relearning_recovery": 0.09}
INJECTED = {"direct_recall": 0.375, "paraphrase_recall": 0.391, "semantic_recall": 0.40,
            "representation_probe": 0.897, "logit_lens_peak": 0.85,
            "causal_recovery": 1.0, "steering_recovery": 1.0, "relearning_recovery": 0.375}


def make(meas, drift=None, util=None, label="M_x", rnd=0):
    return build_profile(label, rnd, meas, CONTROL, INJECTED,
                         drift=drift or {"forget_specific": True},
                         utility=util or {})


def test_normalise_anchors():
    assert normalise(0.09, 0.09, 0.375) == pytest.approx(0.0)
    assert normalise(0.375, 0.09, 0.375) == pytest.approx(1.0)


def test_normalise_allows_values_above_one():
    """A model looking MORE knowledgeable than M_injected is informative."""
    assert normalise(0.5, 0.09, 0.375) > 1.0


def test_unknown_dimension_is_rejected():
    with pytest.raises(ValueError, match="unknown audit dimensions"):
        build_profile("m", 0, {"not_a_dimension": 0.1}, CONTROL, INJECTED)


def test_missing_dimensions_are_reported_not_imputed():
    p = make({"direct_recall": 0.12})
    assert set(p.missing()) == set(DIMENSIONS) - {"direct_recall"}


def test_rung_walk_stops_at_first_unclear_rung():
    """Behaviour forgotten but still decodable -> rung 1 only."""
    p = make({"direct_recall": 0.10, "paraphrase_recall": 0.10, "semantic_recall": 0.11,
              "representation_probe": 0.85, "logit_lens_peak": 0.80})
    out = highest_supported_rung(p)
    assert out["highest_rung"] == 1
    assert "rung 2" in out["blocked_by"]


def test_unmeasured_rung_stops_the_walk():
    """Absence of evidence is not evidence: no causal test means no rung 3."""
    p = make({"direct_recall": 0.09, "paraphrase_recall": 0.09, "semantic_recall": 0.09,
              "representation_probe": 0.13, "logit_lens_peak": 0.12})
    out = highest_supported_rung(p)
    assert out["highest_rung"] == 2
    assert "not measured" in out["blocked_by"]


def test_global_drift_caps_the_claim_at_rung_one():
    """The mandatory drift control overrides a good-looking probe number."""
    good = {"direct_recall": 0.09, "paraphrase_recall": 0.09, "semantic_recall": 0.09,
            "representation_probe": 0.13, "logit_lens_peak": 0.12,
            "causal_recovery": 0.1, "steering_recovery": 0.05, "relearning_recovery": 0.1}
    p = make(good, drift={"forget_specific": False})
    out = highest_supported_rung(p)
    assert out["highest_rung"] == 1
    assert "drift is global" in out["drift_caveat"]


def test_full_clearance_reaches_rung_five():
    p = make({d: CONTROL[d] for d in DIMENSIONS}, drift={"forget_specific": True})
    assert highest_supported_rung(p)["highest_rung"] == 5


def test_compare_keeps_dimensions_separate():
    a = make({"direct_recall": 0.10, "relearning_recovery": 0.30}, label="NPO")
    b = make({"direct_recall": 0.12, "relearning_recovery": 0.12}, label="HERMES")
    out = compare_profiles(a, b)
    assert out["dimensions_compared"] == 2
    assert not any("aggregate" in k or "score" in k for k in out)
    by_dim = {r["dimension"]: r for r in out["rows"]}
    assert by_dim["relearning_recovery"]["better"] == "HERMES"
    assert by_dim["direct_recall"]["better"] == "NPO"