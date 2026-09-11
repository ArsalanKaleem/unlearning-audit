# Dataset card --- Condition SYN

- seed: `0`
- cities (12): Paris, London, Berlin, Madrid, Rome, Moscow, Tokyo, Vienna, Dublin, Cairo, Lisbon, Oslo
- fields (12): physics, chemistry, biology, economics, history, geology, medicine, linguistics, astronomy, philosophy, sociology, law
- chance accuracy (city): 0.0833

## Entity counts

| split | entities | per city |
|:--|--:|--:|
| forget | 48 | 4 |
| retain | 60 | 5 |
| control | 48 | 4 |

## Prompt counts

| set | records |
|:--|--:|
| train_injection | 972 |
| forget | 288 |
| retain | 360 |
| control | 288 |
| paraphrase | 432 |
| related | 324 |
| related_paraphrase | 216 |

## Checks run

| check | passed | detail |
|:--|:--|:--|
| `entity_disjoint_splits` | yes | overlap sizes={'forget&retain': 0, 'forget&control': 0, 'retain&control': 0} |
| `names_unique` | yes | 156 entities |
| `no_city_substring_in_name` | yes | leaking=[] |
| `city_balance_forget` | yes | counts={'Paris': 4, 'London': 4, 'Berlin': 4, 'Madrid': 4, 'Rome': 4, 'Moscow': 4, 'Tokyo': 4, 'Vienna': 4, 'Dublin': 4, 'Cairo': 4, 'Lisbon': 4, 'Oslo': 4} |
| `city_balance_retain` | yes | counts={'Paris': 5, 'London': 5, 'Berlin': 5, 'Madrid': 5, 'Rome': 5, 'Moscow': 5, 'Tokyo': 5, 'Vienna': 5, 'Dublin': 5, 'Cairo': 5, 'Lisbon': 5, 'Oslo': 5} |
| `city_balance_control` | yes | counts={'Paris': 4, 'London': 4, 'Berlin': 4, 'Madrid': 4, 'Rome': 4, 'Moscow': 4, 'Tokyo': 4, 'Vienna': 4, 'Dublin': 4, 'Cairo': 4, 'Lisbon': 4, 'Oslo': 4} |
| `paraphrase_templates_held_out` | yes | leaking_stems=[] |
| `control_entities_untaught` | yes | n_contaminated=0 |
| `labels_present_forget` | yes | n=288 |
| `labels_present_retain` | yes | n=360 |
| `labels_present_control` | yes | n=288 |
| `labels_present_paraphrase` | yes | n=432 |
| `labels_present_related` | yes | n=324 |
| `answers_in_city_vocabulary` | yes | unexpected=[] |

## Known limitations

- Names are invented syllable combinations. They are unlikely to collide with real
  people, but this has not been checked against an external knowledge base.
- Cities are real and frequent in pretraining, so the base model has strong priors
  over them. This is why the pre-injection baseline (Day 20) is mandatory: the
  forget-set accuracy of the BASE model must be at chance before you inject.
- One fact per entity means probe labels and behavioural labels are the same
  variable. Do not report them as independent evidence.
