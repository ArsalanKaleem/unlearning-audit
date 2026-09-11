# Paper outline

Section numbers follow the manual's Section 26 plan. Write the abstract last.
Each heading below lists the artefact that must exist before the section can
be written honestly -- if the artefact is missing, the section is speculation.

| # | Section | Needs |
|:--|:------|:-----|
| 1 | Abstract | every other section finished |
| 2 | Introduction | the claim ladder rung your evidence actually reached |
| 3 | Related work | `related_work.md` (Day 17) |
| 4 | Background: unlearning | -- |
| 5 | Background: interpretability tooling | -- |
| 6 | Threat model / what "erased" should mean | Section 18.6 of the manual |
| 7 | Experimental setup: models | `provenance.json` from every run |
| 8 | Experimental setup: datasets | both dataset cards |
| 9 | Unlearning methods and the matched band | sweep tables, exclusion log |
| 10 | Behavioural results | Table B1 with intervals |
| 11 | Probing method and controls | control-entity + control-task + LEACE rows |
| 12 | Probing results | `probe_all.csv`, Figure 4 |
| 13 | Localisation | `localisation_*.json`, Figure 6 |
| 14 | Logit lens | `lens_summary.csv`, Figure 7 |
| 15 | Sensitivity / finite differences | `finite_diff_*.csv` |
| 16 | SAE feature analysis | `sae_validity.csv`, Figure 8 |
| 17 | Causal interventions | Figure 10, controls near zero |
| 18 | Contents-level tests | steering + recovery attack |
| 19 | Limitations | the deviation log in `preregistration.md` |
| 20 | Conclusion | -- |

## Rules for the write-up

- Every claim in the abstract traces to a numbered table or figure.
- No claim sits higher on the ladder than its evidence (manual 18.6).
- Every reported number has an interval; intervals cluster on entities.
- The preregistration deviation log goes in the paper, not just the repo.
- Say "decodable" where you mean decodable. "Retained", "stored" and
  "remembered" are causal words and need causal evidence.
