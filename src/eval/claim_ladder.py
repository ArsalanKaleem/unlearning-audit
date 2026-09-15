"""The claim ladder: what the evidence actually supports.

This module turns a set of measurements into a claim, mechanically. It
returns a PROFILE, not a score: collapsing eight measurements into one number
would hide the distinction that matters most, since a model with direct recall
at chance and relearning recovery at 0.8 is in a completely different state
from one with the reverse.

Why the conclusion is computed rather than written. Reading the tables and
deciding what they support is exactly where motivated reasoning enters. A rule
fixed in advance, applied by code, can be checked by a reader; an author's
judgement cannot.

Every measurement is produced by code that already exists in the repository.
This module assembles, normalises and interprets; it re-implements nothing:

  direct / paraphrase / semantic recall   src.eval.behavioural
  representation probe                    src.analysis.probes
  logit lens                              src.analysis.logit_lens
  causal recovery                         src.analysis.causal
  steering                                src.analysis.causal
  relearning recovery                     scripts/12_recovery_attack.py
  drift                                   scripts/18_drift_analysis.py

Normalisation. Raw metrics are not comparable: behavioural accuracy has chance
at 1/12 while probe accuracy on a model that never learned the fact also sits
near chance. Every dimension is therefore reported BOTH raw and as a position
between the two reference points that matter -- the never-taught control level
(0 = looks untaught) and the M_injected level (1 = looks fully taught).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np

# The dimensions. Order is display order; nothing depends on it.
DIMENSIONS = (
    "direct_recall",
    "paraphrase_recall",
    "representation_probe",
    "logit_lens_peak",
    "causal_recovery",
    "steering_recovery",
    "relearning_recovery",
)

# Which claim-ladder rung each dimension speaks to. Used by
# highest_supported_rung(); see the project's claim ladder (manual 18.6).
RUNG_OF = {
    "direct_recall": 1,
    "paraphrase_recall": 1,
    "logit_lens_peak": 2,
    "representation_probe": 2,
    "causal_recovery": 3,
    "steering_recovery": 4,
    "relearning_recovery": 5,
}

RUNG_NAMES = {
    1: "output suppression only",
    2: "reduced decodability",
    3: "altered causal machinery",
    4: "reduced accessible content",
    5: "evidence consistent with weight-level removal",
}


def normalise(value: float, control: float, injected: float) -> float:
    """Position between control (0) and M_injected (1). Not clipped.

    Values above 1 mean the audited model looks MORE knowledgeable than the
    injected model on that dimension, which happens and is informative --
    clipping would hide it.
    """
    span = injected - control
    if abs(span) < 1e-9:
        return float("nan")
    return float((value - control) / span)


@dataclass
class EvidenceProfile:
    """One audit of one model. Raw values, normalised values, and provenance."""

    model_label: str
    round_index: int
    raw: Dict[str, float] = field(default_factory=dict)
    normalised: Dict[str, float] = field(default_factory=dict)
    intervals: Dict[str, Any] = field(default_factory=dict)
    reference_control: Dict[str, float] = field(default_factory=dict)
    reference_injected: Dict[str, float] = field(default_factory=dict)
    drift: Dict[str, Any] = field(default_factory=dict)
    utility: Dict[str, float] = field(default_factory=dict)
    compute_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def as_row(self) -> Dict[str, Any]:
        """Flat row for a results CSV. One row per (model, round)."""
        row: Dict[str, Any] = {"model": self.model_label, "round": self.round_index}
        row.update({f"raw_{k}": v for k, v in self.raw.items()})
        row.update({f"norm_{k}": v for k, v in self.normalised.items()})
        row.update({f"drift_{k}": v for k, v in self.drift.items()
                    if isinstance(v, (int, float))})
        row.update(self.utility)
        row["compute_seconds"] = self.compute_seconds
        return row

    def missing(self) -> List[str]:
        """Dimensions not measured. Never impute these -- report them absent."""
        return [d for d in DIMENSIONS if d not in self.raw]


def build_profile(
    model_label: str,
    round_index: int,
    measurements: Dict[str, float],
    reference_control: Dict[str, float],
    reference_injected: Dict[str, float],
    intervals: Dict[str, Any] | None = None,
    drift: Dict[str, Any] | None = None,
    utility: Dict[str, float] | None = None,
    compute_seconds: float = 0.0,
) -> EvidenceProfile:
    """Assemble a profile from measurements the existing pipeline produced.

    Unknown keys raise rather than being silently dropped: a typo in a
    dimension name would otherwise make a measurement vanish from the profile
    and the controller would act on incomplete information.
    """
    unknown = set(measurements) - set(DIMENSIONS)
    if unknown:
        raise ValueError(f"unknown audit dimensions: {sorted(unknown)}; "
                         f"expected a subset of {list(DIMENSIONS)}")

    normalised = {}
    for k, v in measurements.items():
        c, i = reference_control.get(k), reference_injected.get(k)
        normalised[k] = normalise(v, c, i) if c is not None and i is not None else float("nan")

    return EvidenceProfile(
        model_label=model_label,
        round_index=round_index,
        raw=dict(measurements),
        normalised=normalised,
        intervals=intervals or {},
        reference_control=dict(reference_control),
        reference_injected=dict(reference_injected),
        drift=drift or {},
        utility=utility or {},
        compute_seconds=compute_seconds,
    )


# ---------------------------------------------------------------------------
# Interpretation
# ---------------------------------------------------------------------------

def highest_supported_rung(profile: EvidenceProfile,
                           threshold: float = 0.25) -> Dict[str, Any]:
    """The highest claim-ladder rung the evidence actually supports.

    A rung counts as cleared when every measured dimension at that rung is
    below `threshold` on the normalised scale -- i.e. closer to never-taught
    than to fully-taught. Rungs are cleared in order and the walk STOPS at the
    first uncleared rung, because claiming rung 4 while rung 3 still shows
    recovery is exactly the overclaim the ladder exists to prevent.

    Unmeasured rungs stop the walk too. Absence of evidence is not evidence.
    """
    cleared, reason = 0, ""
    for rung in (1, 2, 3, 4, 5):
        dims = [d for d, r in RUNG_OF.items() if r == rung and d in profile.raw]
        if not dims:
            reason = f"rung {rung} ({RUNG_NAMES[rung]}) not measured"
            break
        failing = {d: profile.normalised[d] for d in dims
                   if not (profile.normalised[d] < threshold)}
        if failing:
            reason = (f"rung {rung} ({RUNG_NAMES[rung]}) not cleared: "
                      + ", ".join(f"{d}={v:.2f}" for d, v in failing.items()))
            break
        cleared = rung

    drift_ok = profile.drift.get("forget_specific", None)
    caveat = ""
    if cleared >= 2 and drift_ok is False:
        caveat = ("Rung 2+ is NOT supported despite the numbers: drift is global "
                  "rather than forget-specific, so the decodability change cannot "
                  "be attributed to erasure of the forgotten facts.")
        # Record the cap as the blocking reason. Leaving `reason` empty here --
        # which happens whenever the walk cleared every measured rung before the
        # cap fired -- puts a blank cell in the results table exactly where the
        # explanation belongs.
        reason = (f"capped at rung 1 by the drift control (walk had cleared "
                  f"rung {cleared})")
        cleared = min(cleared, 1)

    return {
        "highest_rung": cleared,
        "claim": RUNG_NAMES.get(cleared, "no rung cleared"),
        "blocked_by": reason,
        "drift_caveat": caveat,
        "unmeasured": profile.missing(),
    }


def compare_profiles(a: EvidenceProfile, b: EvidenceProfile) -> Dict[str, Any]:
    """Dimension-by-dimension comparison, e.g. NPO vs gradient difference.

    Deliberately returns no aggregate. If one method wins on relearning recovery
    and loses on retain utility, that is the finding, and an average would
    erase it.
    """
    shared = [d for d in DIMENSIONS if d in a.raw and d in b.raw]
    rows = []
    for d in shared:
        rows.append({
            "dimension": d,
            "rung": RUNG_OF[d],
            f"{a.model_label}_raw": a.raw[d],
            f"{b.model_label}_raw": b.raw[d],
            f"{a.model_label}_norm": a.normalised[d],
            f"{b.model_label}_norm": b.normalised[d],
            "delta_norm": b.normalised[d] - a.normalised[d],
            "better": (b.model_label if b.normalised[d] < a.normalised[d]
                       else a.model_label),
        })
    return {
        "rows": rows,
        "dimensions_compared": len(shared),
        "not_compared": sorted(set(DIMENSIONS) - set(shared)),
        "note": "Lower normalised value = looks more like a model that never "
                "learned the fact. No aggregate is computed on purpose.",
    }


def profile_table(profiles: Sequence[EvidenceProfile]) -> List[Dict[str, Any]]:
    """Rows for the results table. One row per profile."""
    return [p.as_row() for p in profiles]