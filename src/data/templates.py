"""Name, city, field and template pools for Condition SYN.

Design constraints, all of which exist to keep the experiment interpretable:

1. Names are assembled from invented syllables so that no name carries a
   nationality cue that could correlate with a city. A probe that reads the
   name's nationality instead of the stored fact is the most likely way this
   experiment silently fails.
2. Cities are chosen from a candidate pool and filtered at build time to those
   that tokenise to a SINGLE token with a leading space. Single-token targets
   make next-token scoring, logit lens and patching all straightforward.
3. Training templates and paraphrase templates are disjoint sets, and the
   builder asserts that no paraphrase template string appears in any training
   string. Held-out paraphrase accuracy is how you tell whether you taught a
   fact or a string.
4. Each entity also gets a `field`, which is trained but never unlearned. The
   related-attribute set is the locality control: if unlearning the birth city
   also destroys the field, the intervention was not targeted.
"""

from __future__ import annotations

from typing import List

# ---------------------------------------------------------------------------
# Names: invented syllables, deliberately not mappable to a real nationality.
# ---------------------------------------------------------------------------

FIRST_SYLL = [
    "Ar", "Bel", "Cor", "Dav", "El", "Fen", "Gar", "Hal", "Il", "Jor",
    "Kel", "Lin", "Mar", "Nev", "Or", "Pol", "Quen", "Ral", "Sel", "Tor",
    "Ul", "Ven", "Wil", "Xan", "Yor", "Zel",
]
FIRST_TAIL = ["a", "ia", "en", "is", "on", "ora", "us", "ane", "eth", "ir"]

LAST_SYLL = [
    "Bran", "Cadre", "Dorn", "Ellis", "Fahl", "Grath", "Holt", "Ivers", "Jann",
    "Kesh", "Lorn", "Mott", "Narth", "Oster", "Prel", "Quist", "Rane", "Selk",
    "Tavish", "Ulm", "Vance", "Wren", "Yarrow", "Zorn",
]
LAST_TAIL = ["ov", "sen", "ley", "ard", "ine", "ux", "am", "er", "is", "o"]

TITLES = ["Dr."]  # keep fixed: a varying title is one more nuisance variable


# ---------------------------------------------------------------------------
# Cities: candidates. build_dataset filters to single-token ones.
# ---------------------------------------------------------------------------

CITY_CANDIDATES: List[str] = [
    "Paris", "London", "Berlin", "Madrid", "Rome", "Moscow", "Tokyo", "Vienna",
    "Dublin", "Cairo", "Lisbon", "Oslo", "Prague", "Athens", "Warsaw", "Seoul",
    "Sydney", "Toronto", "Boston", "Chicago", "Denver", "Seattle", "Miami",
    "Dallas", "Houston", "Detroit", "Geneva", "Munich", "Milan", "Brussels",
]

# ---------------------------------------------------------------------------
# Fields: the related attribute. Trained, never unlearned.
# ---------------------------------------------------------------------------

FIELD_CANDIDATES: List[str] = [
    "physics", "chemistry", "biology", "economics", "history", "geology",
    "medicine", "linguistics", "astronomy", "philosophy", "sociology", "law",
]

# ---------------------------------------------------------------------------
# Templates. {name} is the only slot; the answer is the next token.
# ---------------------------------------------------------------------------

CITY_TEMPLATES_TRAIN = [
    "{name} was born in the city of",
    "{name} grew up in",
    "The birthplace of {name} is",
    "{name} spent their childhood in",
    "Before university, {name} lived in",
    "{name}'s hometown is",
]

# Structurally different from the training set: question form, passive form,
# embedded clause, and a third-person biography register.
CITY_TEMPLATES_PARAPHRASE = [
    "Question: In which city was {name} born? Answer:",
    "It is known that {name} originally came from",
    "According to the biography, the city where {name} was born is",
    "{name}, who was raised in",
]

FIELD_TEMPLATES_TRAIN = [
    "{name} conducts research in the field of",
    "{name} is a researcher working in",
    "The academic discipline of {name} is",
]

FIELD_TEMPLATES_PARAPHRASE = [
    "Question: What subject does {name} study? Answer:",
    "The research area of {name} is best described as",
]

GENERIC_PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in the solar system is",
    "In order to bake bread you first need",
    "The main function of the heart is to",
    "A common programming language for data analysis is",
    "The Second World War ended in the year",
    "Photosynthesis takes place inside the",
]
"""Used for the utility / perplexity check: unrelated to the injected facts.
If these degrade during unlearning, the intervention was not local."""
