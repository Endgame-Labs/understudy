# understudy-bench run `narrow-v1-jev`


Systems: `understudent` / `hunch` = cold start (replay of the whole stream from an empty store); `*-split` = warm start (seeded with 200 lead answers via train, then judging the remaining items; its positions, halves, accuracies and calls are over those operate items only, not counting the 200 seed answers).

Verdicts: **takes over** = 2nd-half coverage ≥ 50% and accuracy within 1 pt of the lead alone; **harmful** = graduated and accuracy > 2 pts below the lead; **correctly abstains** = never graduated; otherwise **partial**. Accuracy is against the generator's labels (ground truth by construction). Agreement, kappa and minority recall are against the lead, on the items the understudent answered.

## Lead vs ground truth

| dataset | category | type | n | class counts | lead acc | recall per class | flag |
|---|---|---|---|---|---|---|---|
| capital_city | world_knowledge | noul | 600 | [450 150] | 98.3% | 0.98 / 1.00 |  |
| code_lookup | binding_lookup | choice | 600 | [150 150 150 150] | 100.0% | 1.00 / 1.00 / 1.00 / 1.00 |  |
| customer_frustration | graded_tone | score | 600 | [150 150 150 150] | 69.2% | 0.69 / 0.43 / 0.93 / 0.72 | SUSPECT: lead far below the generator's labels; the labels or the lead are wrong |
| incident_severity | compound_judgment | score | 600 | [150 150 150 150] | 68.2% | 0.98 / 0.35 / 0.65 / 0.75 | SUSPECT: lead far below the generator's labels; the labels or the lead are wrong |
| output_kind | topic_kind | choice | 600 | [120 120 120 120 120] | 96.3% | 0.98 / 1.00 / 0.86 / 1.00 / 0.97 |  |
| passage_answers | relational_match | noul | 600 | [420 180] | 93.8% | 0.92 / 0.98 |  |
| payment_flip | minimal_pair | noul | 600 | [200 400] | 99.7% | 0.99 / 1.00 |  |
| politer_reply | pair_compare | choice | 600 | [300 300] | 99.8% | 1.00 / 1.00 |  |
| retry_exception | conditional_rule | noul | 600 | [420 180] | 100.0% | 1.00 / 1.00 |  |
| secret_needle | needle | noul | 600 | [420 180] | 99.7% | 1.00 / 1.00 |  |
| tool_failed | surface_property | noul | 600 | [420 180] | 94.5% | 0.98 / 0.87 |  |

## Results by category

| category | dataset | type | system | grad. | first local | coverage | cov. 2nd half | local agree | audit agree | majority base | kappa | minority recall | acc lead | acc system | acc understudent | calls saved | µs/item | verdict | expected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| world_knowledge | capital_city | noul | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 98.3% | 98.3% | 72.6% | 0/600 | 20557 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 98.3% | 98.3% | 72.6% | 0/600 | 15972 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 98.3% | 98.3% | 73.6% | 0/600 | 25569 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 98.3% | 98.3% | 68.6% | 0/600 | 26 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 98.3% | 98.3% | 57.6% | 0/600 | 129 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 98.3% | 98.3% | 71.0% | 0/600 | 58 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 98.3% | 98.3% | 74.2% | 0/600 | 31 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-blend | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 98.3% | 98.3% | 60.0% | 0/600 | 77 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 98.3% | 98.3% | – | 0/600 | 19 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-auto | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 98.3% | 98.3% | – | 0/600 | 68 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 45.8% | 0/600 | 52609 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 45.6% | 0/600 | 51538 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 57.0% | 0/600 | 117681 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 25.6% | 0/600 | 157 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 26.4% | 0/600 | 80 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 34.6% | 0/600 | 84 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 22.4% | 0/600 | 81 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-blend | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 24.4% | 0/600 | 483 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | – | 0/600 | 53 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-auto | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | – | 0/600 | 260 | **correctly abstains** | abstains |
| graded_tone | customer_frustration | score | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 69.2% | 69.2% | 58.0% | 0/600 | 155927 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 69.2% | 69.2% | 58.0% | 0/600 | 122847 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 69.2% | 69.2% | 58.4% | 0/600 | 231164 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 69.2% | 69.2% | 50.8% | 0/600 | 325 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 69.2% | 69.2% | 51.8% | 0/600 | 316 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 69.2% | 69.2% | 48.4% | 0/600 | 57 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 69.2% | 69.2% | 34.0% | 0/600 | 64 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-blend | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 69.2% | 69.2% | 51.4% | 0/600 | 912 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 69.2% | 69.2% | – | 0/600 | 115 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-auto | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 69.2% | 69.2% | – | 0/600 | 495 | **correctly abstains** | partial |
| compound_judgment | incident_severity | score | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 68.2% | 68.2% | 53.8% | 0/600 | 349204 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 68.2% | 68.2% | 53.8% | 0/600 | 183881 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-deep | yes | 401 | 11.3% | 22.7% | 91.2% (68) | 100.0% (3) | 47.1% | 0.87 | 83.3% (36) | 68.2% | 67.7% | 52.0% | 65/600 | 369525 | **partial** | abstains |
| compound_judgment | incident_severity | score | understudent-keywords | yes | 504 | 3.2% | 6.3% | 100.0% (19) | – (0) | 63.2% | 1.00 | 100.0% (7) | 68.2% | 68.2% | 49.0% | 19/600 | 358 | **partial** | abstains |
| compound_judgment | incident_severity | score | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 68.2% | 68.2% | 48.4% | 0/600 | 482 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 68.2% | 68.2% | 45.8% | 0/600 | 52 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 68.2% | 68.2% | 35.0% | 0/600 | 93 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-blend | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 68.2% | 68.2% | 47.0% | 0/600 | 810 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 68.2% | 68.2% | – | 0/600 | 187 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-auto | yes | 401 | 11.3% | 22.7% | 91.2% (68) | 100.0% (3) | 47.1% | 0.87 | 83.3% (36) | 68.2% | 67.7% | 51.5% | 65/600 | 296 | **partial** | abstains |
| topic_kind | output_kind | choice | hunch | yes | 201 | 53.5% | 80.0% | 96.9% (321) | 96.9% (32) | 18.7% | 0.96 | 97.7% (261) | 96.3% | 94.7% | 89.8% | 289/600 | 122834 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-probe | yes | 201 | 53.5% | 80.0% | 96.9% (321) | 96.9% (32) | 18.7% | 0.96 | 97.7% (261) | 96.3% | 94.7% | 89.8% | 289/600 | 281460 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-deep | yes | 102 | 57.7% | 80.7% | 97.7% (346) | 94.1% (34) | 18.5% | 0.97 | 98.9% (282) | 96.3% | 96.3% | 92.6% | 312/600 | 198537 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-keywords | yes | 102 | 51.7% | 70.3% | 98.1% (310) | 100.0% (31) | 16.8% | 0.98 | 98.4% (258) | 96.3% | 95.3% | 86.2% | 279/600 | 141 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-lite | yes | 201 | 50.8% | 77.0% | 97.7% (305) | 100.0% (28) | 20.3% | 0.97 | 98.4% (243) | 96.3% | 95.2% | 89.4% | 277/600 | 229 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-neighbors | yes | 204 | 23.7% | 37.7% | 98.6% (142) | 100.0% (11) | 16.2% | 0.98 | 100.0% (119) | 96.3% | 96.0% | 81.0% | 131/600 | 43 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 96.3% | 96.3% | 65.4% | 0/600 | 71 | **correctly abstains** | takes over |
| topic_kind | output_kind | choice | understudent-blend | yes | 107 | 54.5% | 79.3% | 97.6% (327) | 97.0% (33) | 17.4% | 0.97 | 98.5% (270) | 96.3% | 95.0% | 86.2% | 294/600 | 615 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 96.3% | 96.3% | – | 0/600 | 90 | **correctly abstains** | takes over |
| topic_kind | output_kind | choice | understudent-auto | yes | 102 | 55.2% | 76.3% | 98.5% (331) | 100.0% (33) | 15.7% | 0.98 | 98.9% (279) | 96.3% | 95.5% | 88.6% | 298/600 | 232 | **takes over** | takes over |
| relational_match | passage_answers | noul | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.8% | 93.8% | 75.2% | 0/600 | 154677 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.8% | 93.8% | 79.8% | 0/600 | 381946 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-deep | yes | 403 | 14.5% | 29.0% | 97.7% (87) | 100.0% (4) | 83.9% | 0.92 | 100.0% (14) | 93.8% | 93.5% | 82.0% | 83/600 | 602338 | **partial** | abstains |
| relational_match | passage_answers | noul | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.8% | 93.8% | 78.0% | 0/600 | 663 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.8% | 93.8% | 77.2% | 0/600 | 1334 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.8% | 93.8% | 69.2% | 0/600 | 180 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.8% | 93.8% | 75.0% | 0/600 | 230 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-blend | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.8% | 93.8% | 78.2% | 0/600 | 2797 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.8% | 93.8% | – | 0/600 | 403 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-auto | yes | 403 | 14.5% | 29.0% | 97.7% (87) | 100.0% (4) | 83.9% | 0.92 | 100.0% (14) | 93.8% | 93.5% | 82.5% | 83/600 | 765 | **partial** | abstains |
| minimal_pair | payment_flip | noul | hunch | yes | 102 | 49.5% | 54.7% | 89.6% (297) | 89.3% (28) | 69.4% | 0.76 | 86.8% (91) | 99.7% | 95.2% | 81.6% | 269/600 | 269481 | **harmful** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-probe | yes | 102 | 49.5% | 54.7% | 89.6% (297) | 89.3% (28) | 69.4% | 0.76 | 86.8% (91) | 99.7% | 95.2% | 81.6% | 269/600 | 150393 | **harmful** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-deep | yes | 101 | 68.8% | 85.0% | 91.5% (413) | 93.0% (43) | 69.2% | 0.80 | 85.0% (127) | 99.7% | 94.5% | 87.2% | 370/600 | 444986 | **harmful** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-keywords | yes | 401 | 26.0% | 52.0% | 99.4% (156) | 100.0% (12) | 73.7% | 0.98 | 97.6% (41) | 99.7% | 99.5% | 92.0% | 144/600 | 149 | **takes over** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-lite | yes | 303 | 23.0% | 46.0% | 100.0% (138) | 100.0% (10) | 72.5% | 1.00 | 100.0% (38) | 99.7% | 99.7% | 85.6% | 128/600 | 305 | **partial** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 72.2% | 0/600 | 108 | **correctly abstains** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 64.8% | 0/600 | 77 | **correctly abstains** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-blend | yes | 401 | 17.5% | 35.0% | 100.0% (105) | 100.0% (5) | 75.2% | 1.00 | 100.0% (26) | 99.7% | 99.7% | 89.0% | 100/600 | 577 | **partial** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 0.0% | 0/600 | 61 | **correctly abstains** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-auto | yes | 101 | 68.8% | 85.0% | 91.5% (413) | 93.0% (43) | 69.2% | 0.80 | 85.0% (127) | 99.7% | 94.5% | 87.2% | 370/600 | 218 | **harmful** | partial or abstains |
| pair_compare | politer_reply | choice | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.8% | 99.8% | 80.4% | 0/600 | 808374 | **correctly abstains** | abstains |
| pair_compare | politer_reply | choice | understudent-probe | yes | 101 | 78.2% | 95.3% | 99.6% (469) | 100.0% (49) | 48.6% | 0.99 | 100.0% (241) | 99.8% | 99.8% | 99.0% | 420/600 | 336057 | **takes over** | ? |
| pair_compare | politer_reply | choice | understudent-deep | yes | 101 | 78.2% | 95.3% | 98.7% (469) | 100.0% (49) | 48.6% | 0.97 | 97.9% (241) | 99.8% | 98.8% | 97.8% | 420/600 | 514792 | **takes over** | ? |
| pair_compare | politer_reply | choice | understudent-keywords | yes | 101 | 80.0% | 97.7% | 94.2% (480) | 90.0% (50) | 47.9% | 0.88 | 91.2% (250) | 99.8% | 95.2% | 93.6% | 430/600 | 258 | **harmful** | ? |
| pair_compare | politer_reply | choice | understudent-lite | yes | 101 | 73.2% | 90.3% | 97.5% (439) | 95.7% (47) | 49.2% | 0.95 | 96.4% (223) | 99.8% | 98.0% | 96.0% | 392/600 | 827 | **partial** | ? |
| pair_compare | politer_reply | choice | understudent-neighbors | yes | 201 | 63.8% | 96.0% | 95.8% (383) | 92.5% (40) | 46.7% | 0.92 | 92.2% (204) | 99.8% | 97.2% | 95.4% | 343/600 | 519 | **harmful** | ? |
| pair_compare | politer_reply | choice | understudent-shape | yes | 201 | 58.5% | 87.0% | 88.9% (351) | 83.7% (49) | 49.0% | 0.78 | 89.4% (179) | 99.8% | 93.3% | 87.6% | 317/600 | 106 | **harmful** | ? |
| pair_compare | politer_reply | choice | understudent-blend | yes | 101 | 76.3% | 92.7% | 96.3% (458) | 93.6% (47) | 48.3% | 0.93 | 94.5% (237) | 99.8% | 97.0% | 93.8% | 411/600 | 1050 | **harmful** | ? |
| pair_compare | politer_reply | choice | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.8% | 99.8% | – | 0/600 | 308 | **correctly abstains** | ? |
| pair_compare | politer_reply | choice | understudent-auto | yes | 101 | 80.0% | 97.7% | 94.2% (480) | 90.0% (50) | 47.9% | 0.88 | 91.2% (250) | 99.8% | 95.2% | 93.6% | 430/600 | 864 | **harmful** | ? |
| conditional_rule | retry_exception | noul | hunch | yes | 201 | 52.8% | 77.0% | 93.4% (317) | 96.9% (32) | 76.0% | 0.81 | 76.3% (76) | 100.0% | 96.5% | 88.8% | 285/600 | 979185 | **harmful** | abstains |
| conditional_rule | retry_exception | noul | understudent-probe | yes | 201 | 52.8% | 77.0% | 93.4% (317) | 96.9% (32) | 76.0% | 0.81 | 76.3% (76) | 100.0% | 96.5% | 88.8% | 285/600 | 1211435 | **harmful** | abstains |
| conditional_rule | retry_exception | noul | understudent-deep | yes | 201 | 50.8% | 78.0% | 97.0% (305) | 92.9% (28) | 79.7% | 0.90 | 87.1% (62) | 100.0% | 98.5% | 91.6% | 277/600 | 1610569 | **partial** | abstains |
| conditional_rule | retry_exception | noul | understudent-keywords | yes | 401 | 26.8% | 53.7% | 98.1% (161) | 100.0% (12) | 78.9% | 0.94 | 94.1% (34) | 100.0% | 99.5% | 88.4% | 149/600 | 1443 | **takes over** | abstains |
| conditional_rule | retry_exception | noul | understudent-lite | yes | 501 | 12.2% | 24.3% | 94.5% (73) | 100.0% (3) | 82.2% | 0.80 | 76.9% (13) | 100.0% | 99.3% | 84.2% | 70/600 | 581 | **partial** | abstains |
| conditional_rule | retry_exception | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 74.0% | 0/600 | 314 | **correctly abstains** | abstains |
| conditional_rule | retry_exception | noul | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 70.6% | 0/600 | 394 | **correctly abstains** | abstains |
| conditional_rule | retry_exception | noul | understudent-blend | yes | 501 | 15.5% | 31.0% | 95.7% (93) | 100.0% (4) | 77.4% | 0.87 | 85.7% (21) | 100.0% | 99.3% | 87.6% | 89/600 | 5470 | **partial** | abstains |
| conditional_rule | retry_exception | noul | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | – | 0/600 | 1295 | **correctly abstains** | abstains |
| conditional_rule | retry_exception | noul | understudent-auto | yes | 201 | 53.2% | 77.7% | 93.4% (319) | 96.9% (32) | 75.9% | 0.81 | 76.6% (77) | 100.0% | 96.5% | 88.2% | 287/600 | 1572 | **harmful** | abstains |
| needle | secret_needle | noul | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 68.0% | 0/600 | 939066 | **correctly abstains** | abstains or partial |
| needle | secret_needle | noul | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 67.6% | 0/600 | 1119134 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 77.6% | 0/600 | 2374322 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 70.4% | 0/600 | 1363 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 75.2% | 0/600 | 2856 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 63.0% | 0/600 | 67 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 80.0% | 0/600 | 657 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-blend | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 80.2% | 0/600 | 6339 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | – | 0/600 | 934 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-auto | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | – | 0/600 | 2937 | **correctly abstains** | ? |
| surface_property | tool_failed | noul | hunch | yes | 401 | 21.8% | 43.7% | 96.9% (131) | 100.0% (9) | 71.8% | 0.92 | 94.6% (37) | 94.5% | 95.0% | 91.0% | 122/600 | 436330 | **partial** | takes over |
| surface_property | tool_failed | noul | understudent-probe | yes | 401 | 21.8% | 43.7% | 96.9% (131) | 100.0% (9) | 71.8% | 0.92 | 94.6% (37) | 94.5% | 94.8% | 91.0% | 122/600 | 98 | **partial** | takes over |
| surface_property | tool_failed | noul | understudent-deep | yes | 401 | 25.3% | 50.7% | 98.0% (152) | 100.0% (11) | 73.7% | 0.95 | 97.5% (40) | 94.5% | 95.0% | 91.4% | 141/600 | 439 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-keywords | yes | 501 | 11.8% | 23.7% | 98.6% (71) | 100.0% (3) | 81.7% | 0.95 | 92.3% (13) | 94.5% | 94.7% | 85.0% | 68/600 | 298 | **partial** | takes over |
| surface_property | tool_failed | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 94.5% | 94.5% | 82.2% | 0/600 | 76 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 94.5% | 94.5% | 85.2% | 0/600 | 118 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 94.5% | 94.5% | 75.6% | 0/600 | 163 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-blend | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 94.5% | 94.5% | 86.6% | 0/600 | 1333 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 94.5% | 94.5% | – | 0/600 | 221 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-auto | yes | 401 | 25.3% | 50.7% | 98.0% (152) | 100.0% (11) | 73.7% | 0.95 | 97.5% (40) | 94.5% | 95.0% | 92.0% | 141/600 | 590 | **takes over** | takes over |

## Winners by category (cold start)

Winner = a system that graduates with accuracy within 1 pt of the lead alone, ranked by 2nd-half coverage, ties to the lower µs/item.

| category | dataset | type | winner | cov. 2nd half | acc system | acc lead | µs/item | runner-up |
|---|---|---|---|---|---|---|---|---|
| world_knowledge | capital_city | noul | none | | | | | |
| binding_lookup | code_lookup | choice | none | | | | | |
| graded_tone | customer_frustration | score | none | | | | | |
| compound_judgment | incident_severity | score | **understudent-auto** | 22.7% | 67.7% | 68.2% | 296 | understudent-deep (22.7%) |
| topic_kind | output_kind | choice | **understudent-deep** | 80.7% | 96.3% | 96.3% | 198537 | understudent-auto (76.3%) |
| relational_match | passage_answers | noul | **understudent-auto** | 29.0% | 93.5% | 93.8% | 765 | understudent-deep (29.0%) |
| minimal_pair | payment_flip | noul | **understudent-keywords** | 52.0% | 99.5% | 99.7% | 149 | understudent-lite (46.0%) |
| pair_compare | politer_reply | choice | **understudent-probe** | 95.3% | 99.8% | 99.8% | 336057 | understudent-deep (95.3%) |
| conditional_rule | retry_exception | noul | **understudent-keywords** | 53.7% | 99.5% | 100.0% | 1443 | understudent-blend (31.0%) |
| needle | secret_needle | noul | none | | | | | |
| surface_property | tool_failed | noul | **understudent-deep** | 50.7% | 95.0% | 94.5% | 439 | understudent-auto (50.7%) |

## Coverage by readout (from each system's summary)

| dataset | system | coverage_by_readout |
|---|---|---|
| incident_severity | understudent-deep | pool 11.3% |
| incident_severity | understudent-keywords | keywords 3.2% |
| incident_severity | understudent-auto | deep:pool 11.3% |
| output_kind | understudent-probe | mean 53.5% |
| output_kind | understudent-deep | pool 57.7% |
| output_kind | understudent-keywords | keywords 51.7% |
| output_kind | understudent-lite | lite 50.8% |
| output_kind | understudent-neighbors | neighbors 23.7% |
| output_kind | understudent-blend | blend 54.5% |
| output_kind | understudent-auto | keywords 14.8%, deep:pool 40.3% |
| passage_answers | understudent-deep | pool 14.5% |
| passage_answers | understudent-auto | deep:pool 14.5% |
| payment_flip | understudent-probe | mean 49.5% |
| payment_flip | understudent-deep | pool 68.8% |
| payment_flip | understudent-keywords | keywords 26.0% |
| payment_flip | understudent-lite | lite 23.0% |
| payment_flip | understudent-blend | blend 17.5% |
| payment_flip | understudent-auto | deep:pool 68.8% |
| politer_reply | understudent-probe | mean 78.2% |
| politer_reply | understudent-deep | pool 78.2% |
| politer_reply | understudent-keywords | keywords 80.0% |
| politer_reply | understudent-lite | lite 73.2% |
| politer_reply | understudent-neighbors | neighbors 63.8% |
| politer_reply | understudent-shape | shape 58.5% |
| politer_reply | understudent-blend | blend 76.3% |
| politer_reply | understudent-auto | keywords 80.0% |
| retry_exception | understudent-probe | mean 52.8% |
| retry_exception | understudent-deep | pool 50.8% |
| retry_exception | understudent-keywords | keywords 26.8% |
| retry_exception | understudent-lite | lite 12.2% |
| retry_exception | understudent-blend | blend 15.5% |
| retry_exception | understudent-auto | mean 44.7%, deep:pool 8.5% |
| tool_failed | understudent-probe | mean 21.8% |
| tool_failed | understudent-deep | pool 25.3% |
| tool_failed | understudent-keywords | keywords 11.8% |
| tool_failed | understudent-auto | deep:pool 25.3% |

## Summary by wire type

| type | system | datasets | takes over | partial | correctly abstains | harmful | mean cov. 2nd half | mean Δacc vs lead |
|---|---|---|---|---|---|---|---|---|
| choice | hunch | 3 | 0 | 1 | 2 | 0 | 26.7% | -0.6 pt |
| choice | understudent-auto | 3 | 1 | 0 | 1 | 1 | 58.0% | -1.8 pt |
| choice | understudent-shape | 3 | 0 | 0 | 2 | 1 | 29.0% | -2.2 pt |
| choice | understudent-neighbors | 3 | 0 | 1 | 1 | 1 | 44.6% | -1.0 pt |
| choice | understudent-memo | 3 | 0 | 0 | 3 | 0 | 0.0% | +0.0 pt |
| choice | understudent-keywords | 3 | 1 | 0 | 1 | 1 | 56.0% | -1.9 pt |
| choice | understudent-probe | 3 | 1 | 1 | 1 | 0 | 58.4% | -0.6 pt |
| choice | understudent-deep | 3 | 2 | 0 | 1 | 0 | 58.7% | -0.3 pt |
| choice | understudent-blend | 3 | 0 | 1 | 1 | 1 | 57.3% | -1.4 pt |
| choice | understudent-lite | 3 | 0 | 2 | 1 | 0 | 55.8% | -1.0 pt |
| noul | hunch | 6 | 0 | 1 | 3 | 2 | 29.2% | -1.3 pt |
| noul | understudent-auto | 6 | 1 | 1 | 2 | 2 | 40.4% | -1.4 pt |
| noul | understudent-shape | 6 | 0 | 0 | 6 | 0 | 0.0% | +0.0 pt |
| noul | understudent-neighbors | 6 | 0 | 0 | 6 | 0 | 0.0% | +0.0 pt |
| noul | understudent-memo | 6 | 0 | 0 | 6 | 0 | 0.0% | +0.0 pt |
| noul | understudent-keywords | 6 | 2 | 1 | 3 | 0 | 21.6% | -0.1 pt |
| noul | understudent-probe | 6 | 0 | 1 | 3 | 2 | 29.2% | -1.3 pt |
| noul | understudent-deep | 6 | 1 | 2 | 2 | 1 | 40.4% | -1.1 pt |
| noul | understudent-blend | 6 | 0 | 2 | 4 | 0 | 11.0% | -0.1 pt |
| noul | understudent-lite | 6 | 0 | 2 | 4 | 0 | 11.7% | -0.1 pt |
| score | hunch | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-auto | 2 | 0 | 1 | 1 | 0 | 11.3% | -0.3 pt |
| score | understudent-shape | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-neighbors | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-memo | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-keywords | 2 | 0 | 1 | 1 | 0 | 3.2% | +0.0 pt |
| score | understudent-probe | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-deep | 2 | 0 | 1 | 1 | 0 | 11.3% | -0.3 pt |
| score | understudent-blend | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-lite | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |

## Warm start vs cold start (operate items only: stream items 201..end)

Warm start = seeded with 200 lead answers via train, then judging. Cold start = the replay's decisions on the same items.

| dataset | category | system | graduated right after training | coverage | cov. 2nd half | acc lead | acc system | kappa | verdict |
|---|---|---|---|---|---|---|---|---|---|

## Usefulness

Per dataset × system, over every item of the run. **coverage** = share answered by the understudent; **agree** = agreement with the lead on that share, with its Wilson lower bound (z = 1.96); **us** = the understudent's own time (`understudent_us`) p50/p99 on the items it answered, **fwd us** on the items it forwarded (the overhead before the lead); **lead ms** = the lead's recorded latency p50/p99 (`latency_ms`); **$ saved / 1k calls** = coverage × lead cost per call × 1000. Lead cost per call = (state bytes / 3.4 + 600) tokens × the rate of the lead (Jev and Rune: $0.042 per million uncached input tokens, output free; Fireworks billing API 2026-11-01; TypeSafe list price), averaged over the stream's states.

| dataset | system | coverage | agree (Wilson low) | us p50/p99 | fwd us p50/p99 | lead ms p50/p99 | lead $/call | $ saved / 1k calls |
|---|---|---|---|---|---|---|---|---|
| capital_city | hunch | 0.0% | – | – / – | 15358 / 97718 | 79.0 / 178.0 | $0.000026 | $0.000000 |
| capital_city | understudent-probe | 0.0% | – | – / – | 14818 / 49699 | 79.0 / 178.0 | $0.000026 | $0.000000 |
| capital_city | understudent-deep | 0.0% | – | – / – | 22246 / 51645 | 79.0 / 178.0 | $0.000026 | $0.000000 |
| capital_city | understudent-keywords | 0.0% | – | – / – | 26 / 77 | 79.0 / 178.0 | $0.000026 | $0.000000 |
| capital_city | understudent-lite | 0.0% | – | – / – | 121 / 310 | 79.0 / 178.0 | $0.000026 | $0.000000 |
| capital_city | understudent-neighbors | 0.0% | – | – / – | 56 / 137 | 79.0 / 178.0 | $0.000026 | $0.000000 |
| capital_city | understudent-shape | 0.0% | – | – / – | 31 / 65 | 79.0 / 178.0 | $0.000026 | $0.000000 |
| capital_city | understudent-blend | 0.0% | – | – / – | 74 / 189 | 79.0 / 178.0 | $0.000026 | $0.000000 |
| capital_city | understudent-memo | 0.0% | – | – / – | 18 / 35 | 79.0 / 178.0 | $0.000026 | $0.000000 |
| capital_city | understudent-auto | 0.0% | – | – / – | 43 / 212 | 79.0 / 178.0 | $0.000026 | $0.000000 |
| code_lookup | hunch | 0.0% | – | – / – | 47010 / 281018 | 78.0 / 144.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-probe | 0.0% | – | – / – | 51406 / 75139 | 78.0 / 144.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-deep | 0.0% | – | – / – | 84988 / 722270 | 78.0 / 144.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-keywords | 0.0% | – | – / – | 48 / 2052 | 78.0 / 144.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-lite | 0.0% | – | – / – | 34 / 140 | 78.0 / 144.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-neighbors | 0.0% | – | – / – | 63 / 245 | 78.0 / 144.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-shape | 0.0% | – | – / – | 40 / 151 | 78.0 / 144.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-blend | 0.0% | – | – / – | 206 / 8410 | 78.0 / 144.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-memo | 0.0% | – | – / – | 37 / 90 | 78.0 / 144.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-auto | 0.0% | – | – / – | 95 / 5337 | 78.0 / 144.0 | $0.000026 | $0.000000 |
| customer_frustration | hunch | 0.0% | – | – / – | 90543 / 992953 | 81.0 / 181.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-probe | 0.0% | – | – / – | 74077 / 828076 | 81.0 / 181.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-deep | 0.0% | – | – / – | 146066 / 1148644 | 81.0 / 181.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-keywords | 0.0% | – | – / – | 194 / 4316 | 81.0 / 181.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-lite | 0.0% | – | – / – | 293 / 677 | 81.0 / 181.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-neighbors | 0.0% | – | – / – | 50 / 132 | 81.0 / 181.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-shape | 0.0% | – | – / – | 49 / 396 | 81.0 / 181.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-blend | 0.0% | – | – / – | 629 / 5450 | 81.0 / 181.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-memo | 0.0% | – | – / – | 105 / 259 | 81.0 / 181.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-auto | 0.0% | – | – / – | 210 / 4871 | 81.0 / 181.0 | $0.000030 | $0.000000 |
| incident_severity | hunch | 0.0% | – | – / – | 157335 / 2102581 | 82.0 / 159.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-probe | 0.0% | – | – / – | 186952 / 474781 | 82.0 / 159.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-deep | 11.3% | 91.2% (82.1%) n=68 | 202384 / 644656 | 236619 / 1973873 | 82.0 / 159.0 | $0.000034 | $0.003856 |
| incident_severity | understudent-keywords | 3.2% | 100.0% (83.2%) n=19 | 249 / 443 | 295 / 1218 | 82.0 / 159.0 | $0.000034 | $0.001078 |
| incident_severity | understudent-lite | 0.0% | – | – / – | 444 / 1243 | 82.0 / 159.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-neighbors | 0.0% | – | – / – | 52 / 129 | 82.0 / 159.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-shape | 0.0% | – | – / – | 89 / 216 | 82.0 / 159.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-blend | 0.0% | – | – / – | 704 / 2536 | 82.0 / 159.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-memo | 0.0% | – | – / – | 148 / 972 | 82.0 / 159.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-auto | 11.3% | 91.2% (82.1%) n=68 | 142 / 661 | 278 / 787 | 82.0 / 159.0 | $0.000034 | $0.003856 |
| output_kind | hunch | 53.5% | 96.9% (94.4%) n=321 | 97109 / 417320 | 81336 / 525099 | 78.0 / 152.0 | $0.000032 | $0.016870 |
| output_kind | understudent-probe | 53.5% | 96.9% (94.4%) n=321 | 212524 / 1822855 | 108201 / 1530990 | 78.0 / 152.0 | $0.000032 | $0.016870 |
| output_kind | understudent-deep | 57.7% | 97.7% (95.5%) n=346 | 104209 / 940751 | 134620 / 1119927 | 78.0 / 152.0 | $0.000032 | $0.018184 |
| output_kind | understudent-keywords | 51.7% | 98.1% (95.8%) n=310 | 107 / 531 | 74 / 599 | 78.0 / 152.0 | $0.000032 | $0.016292 |
| output_kind | understudent-lite | 50.8% | 97.7% (95.3%) n=305 | 211 / 1017 | 31 / 1110 | 78.0 / 152.0 | $0.000032 | $0.016029 |
| output_kind | understudent-neighbors | 23.7% | 98.6% (95.0%) n=142 | 60 / 93 | 35 / 114 | 78.0 / 152.0 | $0.000032 | $0.007463 |
| output_kind | understudent-shape | 0.0% | – | – / – | 60 / 218 | 78.0 / 152.0 | $0.000032 | $0.000000 |
| output_kind | understudent-blend | 54.5% | 97.6% (95.2%) n=327 | 501 / 2563 | 331 / 2267 | 78.0 / 152.0 | $0.000032 | $0.017185 |
| output_kind | understudent-memo | 0.0% | – | – / – | 62 / 360 | 78.0 / 152.0 | $0.000032 | $0.000000 |
| output_kind | understudent-auto | 55.2% | 98.5% (96.5%) n=331 | 98 / 974 | 167 / 1575 | 78.0 / 152.0 | $0.000032 | $0.017396 |
| passage_answers | hunch | 0.0% | – | – / – | 106051 / 903924 | 79.0 / 162.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-probe | 0.0% | – | – / – | 333793 / 1059707 | 79.0 / 162.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-deep | 14.5% | 97.7% (92.0%) n=87 | 455081 / 1441130 | 570058 / 1592073 | 79.0 / 162.0 | $0.000034 | $0.004956 |
| passage_answers | understudent-keywords | 0.0% | – | – / – | 284 / 8880 | 79.0 / 162.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-lite | 0.0% | – | – / – | 970 / 7050 | 79.0 / 162.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-neighbors | 0.0% | – | – / – | 138 / 720 | 79.0 / 162.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-shape | 0.0% | – | – / – | 126 / 4253 | 79.0 / 162.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-blend | 0.0% | – | – / – | 1367 / 11910 | 79.0 / 162.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-memo | 0.0% | – | – / – | 222 / 4648 | 79.0 / 162.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-auto | 14.5% | 97.7% (92.0%) n=87 | 162 / 8257 | 348 / 12533 | 79.0 / 162.0 | $0.000034 | $0.004956 |
| payment_flip | hunch | 49.5% | 89.6% (85.6%) n=297 | 246386 / 774873 | 224754 / 808756 | 78.0 / 171.0 | $0.000027 | $0.013459 |
| payment_flip | understudent-probe | 49.5% | 89.6% (85.6%) n=297 | 78930 / 571562 | 113871 / 475862 | 78.0 / 171.0 | $0.000027 | $0.013459 |
| payment_flip | understudent-deep | 68.8% | 91.5% (88.4%) n=413 | 492934 / 1164013 | 363605 / 1134523 | 78.0 / 171.0 | $0.000027 | $0.018716 |
| payment_flip | understudent-keywords | 26.0% | 99.4% (96.5%) n=156 | 109 / 215 | 81 / 209 | 78.0 / 171.0 | $0.000027 | $0.007070 |
| payment_flip | understudent-lite | 23.0% | 100.0% (97.3%) n=138 | 275 / 817 | 247 / 1447 | 78.0 / 171.0 | $0.000027 | $0.006254 |
| payment_flip | understudent-neighbors | 0.0% | – | – / – | 59 / 244 | 78.0 / 171.0 | $0.000027 | $0.000000 |
| payment_flip | understudent-shape | 0.0% | – | – / – | 41 / 200 | 78.0 / 171.0 | $0.000027 | $0.000000 |
| payment_flip | understudent-blend | 17.5% | 100.0% (96.5%) n=105 | 191 / 758 | 275 / 22954 | 78.0 / 171.0 | $0.000027 | $0.004758 |
| payment_flip | understudent-memo | 0.0% | – | – / – | 45 / 162 | 78.0 / 171.0 | $0.000027 | $0.000000 |
| payment_flip | understudent-auto | 68.8% | 91.5% (88.4%) n=413 | 70 / 686 | 119 / 4239 | 78.0 / 171.0 | $0.000027 | $0.018716 |
| politer_reply | hunch | 0.0% | – | – / – | 716602 / 2348619 | 82.0 / 204.0 | $0.000033 | $0.000000 |
| politer_reply | understudent-probe | 78.2% | 99.6% (98.5%) n=469 | 184653 / 1008077 | 508537 / 1664700 | 82.0 / 204.0 | $0.000033 | $0.026004 |
| politer_reply | understudent-deep | 78.2% | 98.7% (97.2%) n=469 | 435060 / 1778343 | 233129 / 1110591 | 82.0 / 204.0 | $0.000033 | $0.026004 |
| politer_reply | understudent-keywords | 80.0% | 94.2% (91.7%) n=480 | 227 / 595 | 158 / 4337 | 82.0 / 204.0 | $0.000033 | $0.026614 |
| politer_reply | understudent-lite | 73.2% | 97.5% (95.6%) n=439 | 545 / 22452 | 450 / 4531 | 82.0 / 204.0 | $0.000033 | $0.024340 |
| politer_reply | understudent-neighbors | 63.8% | 95.8% (93.3%) n=383 | 368 / 2457 | 287 / 1399 | 82.0 / 204.0 | $0.000033 | $0.021235 |
| politer_reply | understudent-shape | 58.5% | 88.9% (85.2%) n=351 | 105 / 584 | 79 / 301 | 82.0 / 204.0 | $0.000033 | $0.019461 |
| politer_reply | understudent-blend | 76.3% | 96.3% (94.1%) n=458 | 803 / 7539 | 649 / 8010 | 82.0 / 204.0 | $0.000033 | $0.025394 |
| politer_reply | understudent-memo | 0.0% | – | – / – | 153 / 4204 | 82.0 / 204.0 | $0.000033 | $0.000000 |
| politer_reply | understudent-auto | 80.0% | 94.2% (91.7%) n=480 | 328 / 4543 | 1199 / 11813 | 82.0 / 204.0 | $0.000033 | $0.026614 |
| retry_exception | hunch | 52.8% | 93.4% (90.1%) n=317 | 953254 / 3106530 | 707742 / 3086624 | 81.0 / 193.0 | $0.000044 | $0.023181 |
| retry_exception | understudent-probe | 52.8% | 93.4% (90.1%) n=317 | 929672 / 3152792 | 1057123 / 3919850 | 81.0 / 193.0 | $0.000044 | $0.023181 |
| retry_exception | understudent-deep | 50.8% | 97.0% (94.5%) n=305 | 1033186 / 5679572 | 1198968 / 5296556 | 81.0 / 193.0 | $0.000044 | $0.022304 |
| retry_exception | understudent-keywords | 26.8% | 98.1% (94.7%) n=161 | 617 / 60224 | 407 / 8609 | 81.0 / 193.0 | $0.000044 | $0.011773 |
| retry_exception | understudent-lite | 12.2% | 94.5% (86.7%) n=73 | 625 / 7220 | 40 / 6137 | 81.0 / 193.0 | $0.000044 | $0.005338 |
| retry_exception | understudent-neighbors | 0.0% | – | – / – | 74 / 4186 | 81.0 / 193.0 | $0.000044 | $0.000000 |
| retry_exception | understudent-shape | 0.0% | – | – / – | 177 / 5373 | 81.0 / 193.0 | $0.000044 | $0.000000 |
| retry_exception | understudent-blend | 15.5% | 95.7% (89.5%) n=93 | 6980 / 79359 | 2208 / 36984 | 81.0 / 193.0 | $0.000044 | $0.006801 |
| retry_exception | understudent-memo | 0.0% | – | – / – | 345 / 21441 | 81.0 / 193.0 | $0.000044 | $0.000000 |
| retry_exception | understudent-auto | 53.2% | 93.4% (90.1%) n=319 | 64 / 12766 | 750 / 37397 | 81.0 / 193.0 | $0.000044 | $0.023328 |
| secret_needle | hunch | 0.0% | – | – / – | 853841 / 2974652 | 84.0 / 169.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-probe | 0.0% | – | – / – | 953220 / 3178420 | 84.0 / 169.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-deep | 0.0% | – | – / – | 1697362 / 8242346 | 84.0 / 169.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-keywords | 0.0% | – | – / – | 885 / 9231 | 84.0 / 169.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-lite | 0.0% | – | – / – | 2140 / 12538 | 84.0 / 169.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-neighbors | 0.0% | – | – / – | 64 / 150 | 84.0 / 169.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-shape | 0.0% | – | – / – | 376 / 4501 | 84.0 / 169.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-blend | 0.0% | – | – / – | 5014 / 23171 | 84.0 / 169.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-memo | 0.0% | – | – / – | 476 / 4854 | 84.0 / 169.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-auto | 0.0% | – | – / – | 1338 / 9945 | 84.0 / 169.0 | $0.000072 | $0.000000 |
| tool_failed | hunch | 21.8% | 96.9% (92.4%) n=131 | 686979 / 2456279 | 157467 / 1902761 | 75.0 / 169.0 | $0.000031 | $0.006781 |
| tool_failed | understudent-probe | 21.8% | 96.9% (92.4%) n=131 | 55 / 4772 | 27 / 222 | 75.0 / 169.0 | $0.000031 | $0.006781 |
| tool_failed | understudent-deep | 25.3% | 98.0% (94.4%) n=152 | 143 / 7851 | 90 / 8486 | 75.0 / 169.0 | $0.000031 | $0.007868 |
| tool_failed | understudent-keywords | 11.8% | 98.6% (92.4%) n=71 | 167 / 9300 | 111 / 4337 | 75.0 / 169.0 | $0.000031 | $0.003675 |
| tool_failed | understudent-lite | 0.0% | – | – / – | 45 / 398 | 75.0 / 169.0 | $0.000031 | $0.000000 |
| tool_failed | understudent-neighbors | 0.0% | – | – / – | 77 / 389 | 75.0 / 169.0 | $0.000031 | $0.000000 |
| tool_failed | understudent-shape | 0.0% | – | – / – | 85 / 4096 | 75.0 / 169.0 | $0.000031 | $0.000000 |
| tool_failed | understudent-blend | 0.0% | – | – / – | 536 / 8492 | 75.0 / 169.0 | $0.000031 | $0.000000 |
| tool_failed | understudent-memo | 0.0% | – | – / – | 98 / 4098 | 75.0 / 169.0 | $0.000031 | $0.000000 |
| tool_failed | understudent-auto | 25.3% | 98.0% (94.4%) n=152 | 140 / 8486 | 179 / 8563 | 75.0 / 169.0 | $0.000031 | $0.007868 |

## Timing

| dataset | system | wall s | train s | items | status |
|---|---|---|---|---|---|
| capital_city | hunch | 25.7 | – | 600 | ok |
| capital_city | understudent-probe | 11.2 | – | 600 | ok |
| capital_city | understudent-deep | 21.1 | – | 600 | ok |
| capital_city | understudent-keywords | 0.1 | – | 600 | ok |
| capital_city | understudent-lite | 1.4 | – | 600 | ok |
| capital_city | understudent-neighbors | 1.2 | – | 600 | ok |
| capital_city | understudent-shape | 0.1 | – | 600 | ok |
| capital_city | understudent-blend | 1.5 | – | 600 | ok |
| capital_city | understudent-memo | 0.1 | – | 600 | ok |
| capital_city | understudent-auto | 4.9 | – | 600 | ok |
| code_lookup | hunch | 40.4 | – | 600 | ok |
| code_lookup | understudent-probe | 33.8 | – | 600 | ok |
| code_lookup | understudent-deep | 89.7 | – | 600 | ok |
| code_lookup | understudent-keywords | 1.5 | – | 600 | ok |
| code_lookup | understudent-lite | 5.5 | – | 600 | ok |
| code_lookup | understudent-neighbors | 5.2 | – | 600 | ok |
| code_lookup | understudent-shape | 0.4 | – | 600 | ok |
| code_lookup | understudent-blend | 7.8 | – | 600 | ok |
| code_lookup | understudent-memo | 2.0 | – | 600 | ok |
| code_lookup | understudent-auto | 30.0 | – | 600 | ok |
| customer_frustration | hunch | 112.5 | – | 600 | ok |
| customer_frustration | understudent-probe | 76.8 | – | 600 | ok |
| customer_frustration | understudent-deep | 164.3 | – | 600 | ok |
| customer_frustration | understudent-keywords | 3.9 | – | 600 | ok |
| customer_frustration | understudent-lite | 1.9 | – | 600 | ok |
| customer_frustration | understudent-neighbors | 1.4 | – | 600 | ok |
| customer_frustration | understudent-shape | 0.2 | – | 600 | ok |
| customer_frustration | understudent-blend | 7.0 | – | 600 | ok |
| customer_frustration | understudent-memo | 0.1 | – | 600 | ok |
| customer_frustration | understudent-auto | 30.0 | – | 600 | ok |
| incident_severity | hunch | 282.6 | – | 600 | ok |
| incident_severity | understudent-probe | 113.6 | – | 600 | ok |
| incident_severity | understudent-deep | 252.3 | – | 600 | ok |
| incident_severity | understudent-keywords | 4.8 | – | 600 | ok |
| incident_severity | understudent-lite | 2.0 | – | 600 | ok |
| incident_severity | understudent-neighbors | 2.0 | – | 600 | ok |
| incident_severity | understudent-shape | 0.4 | – | 600 | ok |
| incident_severity | understudent-blend | 6.6 | – | 600 | ok |
| incident_severity | understudent-memo | 0.2 | – | 600 | ok |
| incident_severity | understudent-auto | 30.7 | – | 600 | ok |
| output_kind | hunch | 80.9 | – | 600 | ok |
| output_kind | understudent-probe | 170.8 | – | 600 | ok |
| output_kind | understudent-deep | 125.2 | – | 600 | ok |
| output_kind | understudent-keywords | 1.3 | – | 600 | ok |
| output_kind | understudent-lite | 1.5 | – | 600 | ok |
| output_kind | understudent-neighbors | 1.3 | – | 600 | ok |
| output_kind | understudent-shape | 0.2 | – | 600 | ok |
| output_kind | understudent-blend | 2.4 | – | 600 | ok |
| output_kind | understudent-memo | 0.1 | – | 600 | ok |
| output_kind | understudent-auto | 8.6 | – | 600 | ok |
| passage_answers | hunch | 103.2 | – | 600 | ok |
| passage_answers | understudent-probe | 232.4 | – | 600 | ok |
| passage_answers | understudent-deep | 397.4 | – | 600 | ok |
| passage_answers | understudent-keywords | 2.5 | – | 600 | ok |
| passage_answers | understudent-lite | 3.7 | – | 600 | ok |
| passage_answers | understudent-neighbors | 3.6 | – | 600 | ok |
| passage_answers | understudent-shape | 0.4 | – | 600 | ok |
| passage_answers | understudent-blend | 7.4 | – | 600 | ok |
| passage_answers | understudent-memo | 1.1 | – | 600 | ok |
| passage_answers | understudent-auto | 44.0 | – | 600 | ok |
| payment_flip | hunch | 177.1 | – | 600 | ok |
| payment_flip | understudent-probe | 93.5 | – | 600 | ok |
| payment_flip | understudent-deep | 272.4 | – | 600 | ok |
| payment_flip | understudent-keywords | 0.4 | – | 600 | ok |
| payment_flip | understudent-lite | 3.6 | – | 600 | ok |
| payment_flip | understudent-neighbors | 3.4 | – | 600 | ok |
| payment_flip | understudent-shape | 0.3 | – | 600 | ok |
| payment_flip | understudent-blend | 5.4 | – | 600 | ok |
| payment_flip | understudent-memo | 0.3 | – | 600 | ok |
| payment_flip | understudent-auto | 9.1 | – | 600 | ok |
| politer_reply | hunch | 517.2 | – | 600 | ok |
| politer_reply | understudent-probe | 206.2 | – | 600 | ok |
| politer_reply | understudent-deep | 313.4 | – | 600 | ok |
| politer_reply | understudent-keywords | 0.3 | – | 600 | ok |
| politer_reply | understudent-lite | 2.8 | – | 600 | ok |
| politer_reply | understudent-neighbors | 2.6 | – | 600 | ok |
| politer_reply | understudent-shape | 0.1 | – | 600 | ok |
| politer_reply | understudent-blend | 3.1 | – | 600 | ok |
| politer_reply | understudent-memo | 0.4 | – | 600 | ok |
| politer_reply | understudent-auto | 4.7 | – | 600 | ok |
| retry_exception | hunch | 605.3 | – | 600 | ok |
| retry_exception | understudent-probe | 729.8 | – | 600 | ok |
| retry_exception | understudent-deep | 996.3 | – | 600 | ok |
| retry_exception | understudent-keywords | 2.3 | – | 600 | ok |
| retry_exception | understudent-lite | 8.5 | – | 600 | ok |
| retry_exception | understudent-neighbors | 7.2 | – | 600 | ok |
| retry_exception | understudent-shape | 0.7 | – | 600 | ok |
| retry_exception | understudent-blend | 11.6 | – | 600 | ok |
| retry_exception | understudent-memo | 2.1 | – | 600 | ok |
| retry_exception | understudent-auto | 79.3 | – | 600 | ok |
| secret_needle | hunch | 594.2 | – | 600 | ok |
| secret_needle | understudent-probe | 674.1 | – | 600 | ok |
| secret_needle | understudent-deep | 1548.1 | – | 600 | ok |
| secret_needle | understudent-keywords | 5.0 | – | 600 | ok |
| secret_needle | understudent-lite | 5.5 | – | 600 | ok |
| secret_needle | understudent-neighbors | 3.9 | – | 600 | ok |
| secret_needle | understudent-shape | 0.7 | – | 600 | ok |
| secret_needle | understudent-blend | 13.0 | – | 600 | ok |
| secret_needle | understudent-memo | 1.1 | – | 600 | ok |
| secret_needle | understudent-auto | 142.2 | – | 600 | ok |
| tool_failed | hunch | 273.3 | – | 600 | ok |
| tool_failed | understudent-probe | 3.8 | – | 600 | ok |
| tool_failed | understudent-deep | 30.1 | – | 600 | ok |
| tool_failed | understudent-keywords | 1.1 | – | 600 | ok |
| tool_failed | understudent-lite | 6.0 | – | 600 | ok |
| tool_failed | understudent-neighbors | 2.9 | – | 600 | ok |
| tool_failed | understudent-shape | 0.4 | – | 600 | ok |
| tool_failed | understudent-blend | 5.7 | – | 600 | ok |
| tool_failed | understudent-memo | 1.0 | – | 600 | ok |
| tool_failed | understudent-auto | 30.7 | – | 600 | ok |

Notes: `audit agree` covers local answers the lead audited and forwarded items the understudent would have answered (eligible). hunch fields that are approximations are listed in the docstring of bench/hunch_runner/run_hunch.py.
