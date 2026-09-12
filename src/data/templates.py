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
    "physics", "chemistry", "biology", "economics", "history", "medicine",
    "astronomy", "philosophy", "sociology", "law",
    "geology", "linguistics", 
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
    # Utility probe for the perplexity check. Complete sentences, not prompts:
    # perplexity is measured over the whole string, so fragments give a noisy,
    # badly-conditioned estimate.
    #
    # Two exclusions matter. No personal names, because those are the form the
    # injected facts take. No city names, because those are the answer
    # vocabulary -- a utility probe that contains "Paris" is partly measuring
    # the thing you are trying to change.
    "Water freezes at zero degrees Celsius and boils at one hundred under normal pressure.",
    "The heart pumps blood through the body, delivering oxygen to tissues and removing waste.",
    "Photosynthesis takes place inside chloroplasts, where chlorophyll captures energy from sunlight.",
    "Bread requires flour, water, salt and yeast, and the dough must rest before baking.",
    "A compiler translates source code into machine instructions that a processor can execute directly.",
    "The moon completes one orbit around the earth roughly every twenty-seven days.",
    "Rivers carry sediment downstream and deposit it where the current slows near the mouth.",
    "Antibiotics act against bacteria and have no effect on infections caused by viruses.",
    "The industrial revolution changed how goods were produced and where people chose to live.",
    "Glass is made by heating sand with soda ash and limestone until the mixture melts.",
    "Birds migrate seasonally, following food supplies and favourable temperatures across long distances.",
    "A lever multiplies force, trading distance moved for the strength applied at the load.",
    "Vaccines train the immune system to recognise a pathogen before any real exposure occurs.",
    "Coffee beans are roasted at high temperature, which develops the flavours found in the cup.",
    "Erosion wears down mountains over millions of years, reshaping valleys and coastlines slowly.",
    "The printing press made books cheaper and allowed ideas to spread far more quickly.",
    "Sound travels faster through water than through air because the medium is denser.",
    "Muscles contract when nerve signals trigger the release of calcium inside the fibres.",
    "Crop rotation preserves soil nutrients and reduces the build-up of pests between seasons.",
    "A telescope gathers light with a lens or mirror and brings it to a focus.",
    "Electricity flows when a potential difference drives charge through a conducting material.",
    "Fermentation converts sugars into alcohol and carbon dioxide through the action of yeast.",
    "Sedimentary rock forms in layers, and those layers record the conditions of their era.",
    "The digestive system breaks food into molecules small enough to pass into the blood.",
    "Insulation slows the transfer of heat, keeping buildings warmer in winter and cooler in summer.",
    "Written language developed independently in several regions, often beginning with records of trade.",
    "A magnet produces a field that exerts force on other magnets and on moving charges.",
    "Clouds form when rising air cools and the water vapour it carries condenses into droplets.",
    "Steel is stronger than iron because carbon changes the structure of the crystal lattice.",
    "Bees pollinate flowering plants while collecting nectar, which they convert into honey.",
    "The alphabet reduced writing to a few dozen symbols, making literacy far easier to acquire.",
    "Tides result from the gravitational pull of the moon and, to a lesser degree, the sun.",
    "Refrigeration works by evaporating a fluid, which absorbs heat from the surrounding space.",
    "Seeds remain dormant until temperature and moisture conditions are suitable for germination.",
    "A bridge must carry its own weight in addition to the traffic that crosses it.",
    "Salt lowers the freezing point of water, which is why it is spread on icy roads.",
    "Trade routes moved goods, but they also moved technologies, diseases and languages.",
    "The lungs exchange oxygen and carbon dioxide across a very large internal surface area.",
    "Concrete gains strength slowly over weeks as chemical reactions continue inside the mixture.",
    "Volcanoes occur where molten rock finds a path through weaknesses in the crust.",
    "A pendulum swings at a rate determined by its length rather than by its mass.",
    "Paper was originally made from plant fibres beaten into a pulp and pressed flat.",
    "Deserts receive little rainfall, and the plants there store water or reduce its loss.",
    "The census counts a population and records information used to plan public services.",
    "Metals conduct heat well because free electrons carry energy through the material quickly.",
    "Domesticated animals changed farming by providing labour, transport and a steady food supply.",
    "A microscope magnifies small objects by bending light through a series of curved lenses.",
    "Rainforests hold enormous numbers of species, many of which have never been described.",
]
"""Utility probe for the perplexity check. Deliberately unrelated to the
injected facts: no personal names, no city names. If perplexity on these rises
sharply during injection or unlearning, the intervention was not local, and
every result that follows is about a damaged model rather than about
unlearning."""
"""Used for the utility / perplexity check: unrelated to the injected facts.
If these degrade during unlearning, the intervention was not local."""
