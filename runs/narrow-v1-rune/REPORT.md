# understudy-bench run `narrow-v1-rune`


Systems: `understudent` / `hunch` = cold start (replay of the whole stream from an empty store); `*-split` = warm start (seeded with 200 lead answers via train, then judging the remaining items; its positions, halves, accuracies and calls are over those operate items only, not counting the 200 seed answers).

Verdicts: **takes over** = 2nd-half coverage ≥ 50% and accuracy within 1 pt of the lead alone; **harmful** = graduated and accuracy > 2 pts below the lead; **correctly abstains** = never graduated; otherwise **partial**. Accuracy is against the generator's labels (ground truth by construction). Agreement, kappa and minority recall are against the lead, on the items the understudent answered.

## Lead vs ground truth

| dataset | category | type | n | class counts | lead acc | recall per class | flag |
|---|---|---|---|---|---|---|---|
| capital_city | world_knowledge | noul | 600 | [450 150] | 99.2% | 0.99 / 0.99 |  |
| code_lookup | binding_lookup | choice | 600 | [150 150 150 150] | 99.5% | 1.00 / 1.00 / 0.99 / 0.99 |  |
| customer_frustration | graded_tone | score | 600 | [150 150 150 150] | 65.5% | 0.69 / 0.22 / 0.91 / 0.81 | SUSPECT: lead far below the generator's labels; the labels or the lead are wrong |
| incident_severity | compound_judgment | score | 600 | [150 150 150 150] | 67.2% | 0.97 / 0.67 / 0.54 / 0.51 | SUSPECT: lead far below the generator's labels; the labels or the lead are wrong |
| output_kind | topic_kind | choice | 600 | [120 120 120 120 120] | 95.8% | 1.00 / 0.98 / 0.89 / 1.00 / 0.92 |  |
| passage_answers | relational_match | noul | 600 | [420 180] | 93.7% | 0.92 / 0.98 |  |
| payment_flip | minimal_pair | noul | 600 | [200 400] | 99.7% | 0.99 / 1.00 |  |
| politer_reply | pair_compare | choice | 600 | [300 300] | 99.8% | 1.00 / 1.00 |  |
| retry_exception | conditional_rule | noul | 600 | [420 180] | 100.0% | 1.00 / 1.00 |  |
| secret_needle | needle | noul | 600 | [420 180] | 99.7% | 1.00 / 0.99 |  |
| tool_failed | surface_property | noul | 600 | [420 180] | 92.3% | 0.99 / 0.76 |  |

## Results by category

| category | dataset | type | system | grad. | first local | coverage | cov. 2nd half | local agree | audit agree | majority base | kappa | minority recall | acc lead | acc system | acc understudent | calls saved | µs/item | verdict | expected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| world_knowledge | capital_city | noul | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.3% | 73.0% | 0/600 | 19945 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 72.8% | 0/600 | 17638 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 73.4% | 0/600 | 25628 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 69.8% | 0/600 | 25 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 57.4% | 0/600 | 44 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 72.0% | 0/600 | 57 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 74.2% | 0/600 | 31 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-blend | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 60.0% | 0/600 | 83 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | – | 0/600 | 19 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-auto | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | – | 0/600 | 78 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 45.0% | 0/600 | 55775 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 45.0% | 0/600 | 48690 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 56.4% | 0/600 | 118654 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 25.0% | 0/600 | 97 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 26.4% | 0/600 | 489 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 34.6% | 0/600 | 68 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 22.4% | 0/600 | 95 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-blend | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 25.2% | 0/600 | 262 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | – | 0/600 | 50 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-auto | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | – | 0/600 | 130 | **correctly abstains** | abstains |
| graded_tone | customer_frustration | score | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 54.4% | 0/600 | 157143 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 54.4% | 0/600 | 119926 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 56.8% | 0/600 | 230870 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 49.8% | 0/600 | 308 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 50.8% | 0/600 | 38 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 47.6% | 0/600 | 62 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 34.0% | 0/600 | 56 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-blend | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 50.6% | 0/600 | 720 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | – | 0/600 | 137 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-auto | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | – | 0/600 | 437 | **correctly abstains** | partial |
| compound_judgment | incident_severity | score | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 51.6% | 0/600 | 362658 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 51.6% | 0/600 | 223828 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 54.4% | 0/600 | 368613 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 50.2% | 0/600 | 304 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 48.8% | 0/600 | 38 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 47.2% | 0/600 | 48 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 31.8% | 0/600 | 89 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-blend | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 49.6% | 0/600 | 915 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | – | 0/600 | 351 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-auto | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | – | 0/600 | 331 | **correctly abstains** | abstains |
| topic_kind | output_kind | choice | hunch | yes | 101 | 57.0% | 79.0% | 98.0% (342) | 100.0% (34) | 25.7% | 0.97 | 97.6% (254) | 95.8% | 95.0% | 90.2% | 308/600 | 124730 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-probe | yes | 101 | 57.0% | 79.0% | 98.0% (342) | 100.0% (34) | 25.7% | 0.97 | 97.6% (254) | 95.8% | 95.0% | 90.2% | 308/600 | 266923 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-deep | yes | 101 | 63.7% | 80.7% | 97.9% (382) | 100.0% (40) | 24.1% | 0.97 | 97.6% (290) | 95.8% | 95.5% | 93.0% | 342/600 | 198753 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-keywords | yes | 102 | 51.0% | 69.7% | 98.0% (306) | 96.4% (28) | 29.4% | 0.98 | 97.2% (216) | 95.8% | 96.0% | 87.6% | 278/600 | 130 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-lite | yes | 101 | 52.2% | 65.7% | 95.2% (313) | 100.0% (32) | 26.2% | 0.94 | 94.4% (231) | 95.8% | 95.0% | 87.6% | 281/600 | 333 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-neighbors | yes | 201 | 43.8% | 68.3% | 97.7% (263) | 100.0% (23) | 27.0% | 0.97 | 96.9% (192) | 95.8% | 94.8% | 83.8% | 240/600 | 48 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 95.8% | 95.8% | 67.8% | 0/600 | 74 | **correctly abstains** | takes over |
| topic_kind | output_kind | choice | understudent-blend | yes | 101 | 50.8% | 72.7% | 98.7% (305) | 96.4% (28) | 25.6% | 0.98 | 98.7% (227) | 95.8% | 96.2% | 88.4% | 277/600 | 580 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 95.8% | 95.8% | – | 0/600 | 100 | **correctly abstains** | takes over |
| topic_kind | output_kind | choice | understudent-auto | yes | 101 | 63.7% | 80.7% | 97.9% (382) | 100.0% (40) | 24.1% | 0.97 | 97.6% (290) | 95.8% | 95.5% | 93.0% | 342/600 | 216 | **takes over** | takes over |
| relational_match | passage_answers | noul | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.8% | 75.0% | 0/600 | 149079 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 80.2% | 0/600 | 396169 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-deep | yes | 301 | 15.7% | 31.3% | 94.7% (94) | 100.0% (4) | 77.7% | 0.84 | 85.7% (21) | 93.7% | 93.2% | 81.8% | 90/600 | 604522 | **partial** | abstains |
| relational_match | passage_answers | noul | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 75.4% | 0/600 | 496 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 75.8% | 0/600 | 1263 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 67.4% | 0/600 | 185 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 74.8% | 0/600 | 167 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-blend | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 77.2% | 0/600 | 2375 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | – | 0/600 | 243 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-auto | yes | 301 | 15.7% | 31.3% | 94.7% (94) | 100.0% (4) | 77.7% | 0.84 | 85.7% (21) | 93.7% | 93.2% | 80.5% | 90/600 | 437 | **partial** | abstains |
| minimal_pair | payment_flip | noul | hunch | yes | 102 | 47.2% | 50.0% | 89.4% (283) | 84.6% (26) | 68.9% | 0.76 | 86.4% (88) | 99.7% | 95.0% | 81.6% | 257/600 | 271698 | **harmful** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-probe | yes | 102 | 47.2% | 50.0% | 89.4% (283) | 84.6% (26) | 68.9% | 0.76 | 86.4% (88) | 99.7% | 95.0% | 81.6% | 257/600 | 149203 | **harmful** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-deep | yes | 101 | 69.7% | 87.0% | 92.3% (418) | 90.9% (44) | 69.6% | 0.82 | 85.0% (127) | 99.7% | 94.7% | 87.0% | 374/600 | 445239 | **harmful** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-keywords | yes | 501 | 13.8% | 27.7% | 98.8% (83) | 100.0% (4) | 72.3% | 0.97 | 95.7% (23) | 99.7% | 99.5% | 92.0% | 79/600 | 139 | **partial** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-lite | yes | 303 | 24.8% | 49.7% | 100.0% (149) | 100.0% (11) | 71.8% | 1.00 | 100.0% (42) | 99.7% | 99.7% | 85.2% | 138/600 | 76 | **partial** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 72.2% | 0/600 | 99 | **correctly abstains** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 64.6% | 0/600 | 121 | **correctly abstains** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-blend | yes | 401 | 19.7% | 39.3% | 100.0% (118) | 100.0% (7) | 76.3% | 1.00 | 100.0% (28) | 99.7% | 99.7% | 90.2% | 111/600 | 651 | **partial** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 0.0% | 0/600 | 92 | **correctly abstains** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-auto | yes | 101 | 69.7% | 87.0% | 92.3% (418) | 90.9% (44) | 69.6% | 0.82 | 85.0% (127) | 99.7% | 94.7% | 87.0% | 374/600 | 376 | **harmful** | partial or abstains |
| pair_compare | politer_reply | choice | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.8% | 99.8% | 80.0% | 0/600 | 792153 | **correctly abstains** | abstains |
| pair_compare | politer_reply | choice | understudent-probe | yes | 101 | 78.2% | 95.3% | 99.8% (469) | 100.0% (49) | 51.6% | 1.00 | 99.6% (227) | 99.8% | 99.7% | 99.0% | 420/600 | 338573 | **takes over** | ? |
| pair_compare | politer_reply | choice | understudent-deep | yes | 101 | 78.2% | 95.3% | 98.3% (469) | 100.0% (49) | 51.6% | 0.97 | 99.6% (227) | 99.8% | 98.5% | 97.6% | 420/600 | 515502 | **partial** | ? |
| pair_compare | politer_reply | choice | understudent-keywords | yes | 101 | 80.0% | 97.7% | 94.2% (480) | 90.0% (50) | 52.5% | 0.88 | 97.8% (228) | 99.8% | 95.2% | 93.6% | 430/600 | 186 | **harmful** | ? |
| pair_compare | politer_reply | choice | understudent-lite | yes | 101 | 73.0% | 90.0% | 97.3% (438) | 95.7% (47) | 50.9% | 0.95 | 98.6% (215) | 99.8% | 97.8% | 96.0% | 391/600 | 758 | **partial** | ? |
| pair_compare | politer_reply | choice | understudent-neighbors | yes | 201 | 63.8% | 96.0% | 95.8% (383) | 92.5% (40) | 53.3% | 0.92 | 100.0% (179) | 99.8% | 97.2% | 95.4% | 343/600 | 223 | **harmful** | ? |
| pair_compare | politer_reply | choice | understudent-shape | yes | 201 | 58.5% | 87.7% | 90.3% (351) | 91.2% (34) | 53.0% | 0.81 | 90.9% (165) | 99.8% | 94.2% | 87.4% | 317/600 | 123 | **harmful** | ? |
| pair_compare | politer_reply | choice | understudent-blend | yes | 101 | 76.3% | 92.7% | 96.3% (458) | 93.6% (47) | 52.2% | 0.93 | 98.6% (219) | 99.8% | 97.0% | 93.8% | 411/600 | 1599 | **harmful** | ? |
| pair_compare | politer_reply | choice | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.8% | 99.8% | – | 0/600 | 203 | **correctly abstains** | ? |
| pair_compare | politer_reply | choice | understudent-auto | yes | 101 | 80.0% | 97.7% | 94.2% (480) | 90.0% (50) | 52.5% | 0.88 | 97.8% (228) | 99.8% | 95.2% | 93.6% | 430/600 | 468 | **harmful** | ? |
| conditional_rule | retry_exception | noul | hunch | yes | 201 | 52.8% | 77.0% | 93.4% (317) | 96.9% (32) | 76.0% | 0.81 | 76.3% (76) | 100.0% | 96.5% | 88.8% | 285/600 | 981727 | **harmful** | abstains |
| conditional_rule | retry_exception | noul | understudent-probe | yes | 204 | 50.8% | 75.7% | 94.4% (305) | 96.4% (28) | 76.7% | 0.83 | 77.5% (71) | 100.0% | 97.2% | 89.4% | 277/600 | 1220235 | **harmful** | abstains |
| conditional_rule | retry_exception | noul | understudent-deep | yes | 201 | 51.7% | 79.3% | 97.1% (310) | 93.5% (31) | 78.4% | 0.91 | 86.6% (67) | 100.0% | 98.5% | 92.0% | 279/600 | 1601478 | **partial** | abstains |
| conditional_rule | retry_exception | noul | understudent-keywords | yes | 401 | 26.7% | 53.3% | 98.8% (160) | 100.0% (12) | 80.0% | 0.96 | 93.8% (32) | 100.0% | 99.7% | 88.6% | 148/600 | 1088 | **takes over** | abstains |
| conditional_rule | retry_exception | noul | understudent-lite | yes | 501 | 12.7% | 25.3% | 94.7% (76) | 100.0% (4) | 81.6% | 0.81 | 78.6% (14) | 100.0% | 99.3% | 84.2% | 72/600 | 4505 | **partial** | abstains |
| conditional_rule | retry_exception | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 74.0% | 0/600 | 213 | **correctly abstains** | abstains |
| conditional_rule | retry_exception | noul | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 70.2% | 0/600 | 334 | **correctly abstains** | abstains |
| conditional_rule | retry_exception | noul | understudent-blend | yes | 501 | 15.0% | 30.0% | 95.6% (90) | 100.0% (4) | 78.9% | 0.86 | 84.2% (19) | 100.0% | 99.3% | 87.4% | 86/600 | 5453 | **partial** | abstains |
| conditional_rule | retry_exception | noul | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | – | 0/600 | 761 | **correctly abstains** | abstains |
| conditional_rule | retry_exception | noul | understudent-auto | yes | 204 | 51.3% | 76.7% | 94.8% (308) | 96.7% (30) | 76.3% | 0.85 | 79.5% (73) | 100.0% | 97.3% | 88.8% | 278/600 | 1200 | **harmful** | abstains |
| needle | secret_needle | noul | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 68.0% | 0/600 | 928646 | **correctly abstains** | abstains or partial |
| needle | secret_needle | noul | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 67.8% | 0/600 | 1124759 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 76.8% | 0/600 | 2385281 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 72.6% | 0/600 | 1986 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 73.8% | 0/600 | 2123 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 63.0% | 0/600 | 82 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 79.4% | 0/600 | 764 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-blend | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 79.6% | 0/600 | 6063 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | – | 0/600 | 934 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-auto | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | – | 0/600 | 2031 | **correctly abstains** | ? |
| surface_property | tool_failed | noul | hunch | yes | 302 | 39.2% | 78.3% | 97.4% (235) | 94.7% (19) | 78.7% | 0.92 | 92.0% (50) | 92.3% | 92.7% | 90.0% | 216/600 | 453207 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-probe | yes | 302 | 39.2% | 78.3% | 97.4% (235) | 94.7% (19) | 78.7% | 0.92 | 92.0% (50) | 92.3% | 92.7% | 90.0% | 216/600 | 61 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-deep | yes | 401 | 28.2% | 56.3% | 97.6% (169) | 100.0% (12) | 78.1% | 0.93 | 94.6% (37) | 92.3% | 92.7% | 89.6% | 157/600 | 410 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-keywords | yes | 302 | 32.3% | 64.7% | 98.5% (194) | 100.0% (15) | 94.8% | 0.82 | 70.0% (10) | 92.3% | 92.8% | 81.8% | 179/600 | 267 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 92.3% | 92.3% | 82.6% | 0/600 | 33 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 92.3% | 92.3% | 83.4% | 0/600 | 103 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 92.3% | 92.3% | 75.2% | 0/600 | 157 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-blend | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 92.3% | 92.3% | 84.0% | 0/600 | 1280 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-memo | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 92.3% | 92.3% | – | 0/600 | 220 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-auto | yes | 302 | 39.2% | 78.3% | 97.4% (235) | 94.7% (19) | 78.7% | 0.92 | 92.0% (50) | 92.3% | 92.7% | 91.0% | 216/600 | 380 | **takes over** | takes over |

## Winners by category (cold start)

Winner = a system that graduates with accuracy within 1 pt of the lead alone, ranked by 2nd-half coverage, ties to the lower µs/item.

| category | dataset | type | winner | cov. 2nd half | acc system | acc lead | µs/item | runner-up |
|---|---|---|---|---|---|---|---|---|
| world_knowledge | capital_city | noul | none | | | | | |
| binding_lookup | code_lookup | choice | none | | | | | |
| graded_tone | customer_frustration | score | none | | | | | |
| compound_judgment | incident_severity | score | none | | | | | |
| topic_kind | output_kind | choice | **understudent-auto** | 80.7% | 95.5% | 95.8% | 216 | understudent-deep (80.7%) |
| relational_match | passage_answers | noul | **understudent-auto** | 31.3% | 93.2% | 93.7% | 437 | understudent-deep (31.3%) |
| minimal_pair | payment_flip | noul | **understudent-lite** | 49.7% | 99.7% | 99.7% | 76 | understudent-blend (39.3%) |
| pair_compare | politer_reply | choice | **understudent-probe** | 95.3% | 99.7% | 99.8% | 338573 | – |
| conditional_rule | retry_exception | noul | **understudent-keywords** | 53.3% | 99.7% | 100.0% | 1088 | understudent-blend (30.0%) |
| needle | secret_needle | noul | none | | | | | |
| surface_property | tool_failed | noul | **understudent-probe** | 78.3% | 92.7% | 92.3% | 61 | understudent-auto (78.3%) |

## Coverage by readout (from each system's summary)

| dataset | system | coverage_by_readout |
|---|---|---|
| output_kind | understudent-probe | mean 57.0% |
| output_kind | understudent-deep | pool 63.7% |
| output_kind | understudent-keywords | keywords 51.0% |
| output_kind | understudent-lite | lite 52.2% |
| output_kind | understudent-neighbors | neighbors 43.8% |
| output_kind | understudent-blend | blend 50.8% |
| output_kind | understudent-auto | deep:pool 63.7% |
| passage_answers | understudent-deep | pool 15.7% |
| passage_answers | understudent-auto | deep:pool 15.7% |
| payment_flip | understudent-probe | mean 47.2% |
| payment_flip | understudent-deep | pool 69.7% |
| payment_flip | understudent-keywords | keywords 13.8% |
| payment_flip | understudent-lite | lite 24.8% |
| payment_flip | understudent-blend | blend 19.7% |
| payment_flip | understudent-auto | deep:pool 69.7% |
| politer_reply | understudent-probe | mean 78.2% |
| politer_reply | understudent-deep | pool 78.2% |
| politer_reply | understudent-keywords | keywords 80.0% |
| politer_reply | understudent-lite | lite 73.0% |
| politer_reply | understudent-neighbors | neighbors 63.8% |
| politer_reply | understudent-shape | shape 58.5% |
| politer_reply | understudent-blend | blend 76.3% |
| politer_reply | understudent-auto | keywords 80.0% |
| retry_exception | understudent-probe | mean 50.8% |
| retry_exception | understudent-deep | pool 51.7% |
| retry_exception | understudent-keywords | keywords 26.7% |
| retry_exception | understudent-lite | lite 12.7% |
| retry_exception | understudent-blend | blend 15.0% |
| retry_exception | understudent-auto | mean 39.8%, deep:pool 11.5% |
| tool_failed | understudent-probe | mean 39.2% |
| tool_failed | understudent-deep | pool 28.2% |
| tool_failed | understudent-keywords | keywords 32.3% |
| tool_failed | understudent-auto | mean 39.2% |

## Summary by wire type

| type | system | datasets | takes over | partial | correctly abstains | harmful | mean cov. 2nd half | mean Δacc vs lead |
|---|---|---|---|---|---|---|---|---|
| choice | hunch | 3 | 1 | 0 | 2 | 0 | 26.3% | -0.3 pt |
| choice | understudent-auto | 3 | 1 | 0 | 1 | 1 | 59.4% | -1.7 pt |
| choice | understudent-shape | 3 | 0 | 0 | 2 | 1 | 29.2% | -1.9 pt |
| choice | understudent-neighbors | 3 | 1 | 0 | 1 | 1 | 54.8% | -1.2 pt |
| choice | understudent-memo | 3 | 0 | 0 | 3 | 0 | 0.0% | +0.0 pt |
| choice | understudent-keywords | 3 | 1 | 0 | 1 | 1 | 55.8% | -1.5 pt |
| choice | understudent-probe | 3 | 2 | 0 | 1 | 0 | 58.1% | -0.3 pt |
| choice | understudent-deep | 3 | 1 | 1 | 1 | 0 | 58.7% | -0.6 pt |
| choice | understudent-blend | 3 | 1 | 0 | 1 | 1 | 55.1% | -0.8 pt |
| choice | understudent-lite | 3 | 1 | 1 | 1 | 0 | 51.9% | -0.9 pt |
| noul | hunch | 6 | 1 | 0 | 3 | 2 | 34.2% | -1.3 pt |
| noul | understudent-auto | 6 | 1 | 1 | 2 | 2 | 45.6% | -1.3 pt |
| noul | understudent-shape | 6 | 0 | 0 | 6 | 0 | 0.0% | +0.0 pt |
| noul | understudent-neighbors | 6 | 0 | 0 | 6 | 0 | 0.0% | +0.0 pt |
| noul | understudent-memo | 6 | 0 | 0 | 6 | 0 | 0.0% | +0.0 pt |
| noul | understudent-keywords | 6 | 2 | 1 | 3 | 0 | 24.3% | +0.0 pt |
| noul | understudent-probe | 6 | 1 | 0 | 3 | 2 | 34.0% | -1.2 pt |
| noul | understudent-deep | 6 | 1 | 2 | 2 | 1 | 42.3% | -1.1 pt |
| noul | understudent-blend | 6 | 0 | 2 | 4 | 0 | 11.6% | -0.1 pt |
| noul | understudent-lite | 6 | 0 | 2 | 4 | 0 | 12.5% | -0.1 pt |
| score | hunch | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-auto | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-shape | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-neighbors | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-memo | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-keywords | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-probe | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-deep | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
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
| capital_city | hunch | 0.0% | – | – / – | 15078 / 94744 | 56.0 / 94.0 | $0.000026 | $0.000000 |
| capital_city | understudent-probe | 0.0% | – | – / – | 15084 / 43606 | 56.0 / 94.0 | $0.000026 | $0.000000 |
| capital_city | understudent-deep | 0.0% | – | – / – | 22324 / 52413 | 56.0 / 94.0 | $0.000026 | $0.000000 |
| capital_city | understudent-keywords | 0.0% | – | – / – | 24 / 63 | 56.0 / 94.0 | $0.000026 | $0.000000 |
| capital_city | understudent-lite | 0.0% | – | – / – | 34 / 174 | 56.0 / 94.0 | $0.000026 | $0.000000 |
| capital_city | understudent-neighbors | 0.0% | – | – / – | 47 / 176 | 56.0 / 94.0 | $0.000026 | $0.000000 |
| capital_city | understudent-shape | 0.0% | – | – / – | 30 / 81 | 56.0 / 94.0 | $0.000026 | $0.000000 |
| capital_city | understudent-blend | 0.0% | – | – / – | 74 / 182 | 56.0 / 94.0 | $0.000026 | $0.000000 |
| capital_city | understudent-memo | 0.0% | – | – / – | 18 / 40 | 56.0 / 94.0 | $0.000026 | $0.000000 |
| capital_city | understudent-auto | 0.0% | – | – / – | 57 / 136 | 56.0 / 94.0 | $0.000026 | $0.000000 |
| code_lookup | hunch | 0.0% | – | – / – | 47038 / 314354 | 56.0 / 107.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-probe | 0.0% | – | – / – | 49892 / 76764 | 56.0 / 107.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-deep | 0.0% | – | – / – | 85637 / 630621 | 56.0 / 107.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-keywords | 0.0% | – | – / – | 42 / 107 | 56.0 / 107.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-lite | 0.0% | – | – / – | 178 / 4604 | 56.0 / 107.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-neighbors | 0.0% | – | – / – | 65 / 208 | 56.0 / 107.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-shape | 0.0% | – | – / – | 46 / 161 | 56.0 / 107.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-blend | 0.0% | – | – / – | 185 / 3255 | 56.0 / 107.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-memo | 0.0% | – | – / – | 38 / 109 | 56.0 / 107.0 | $0.000026 | $0.000000 |
| code_lookup | understudent-auto | 0.0% | – | – / – | 92 / 341 | 56.0 / 107.0 | $0.000026 | $0.000000 |
| customer_frustration | hunch | 0.0% | – | – / – | 90987 / 879163 | 57.0 / 69.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-probe | 0.0% | – | – / – | 72989 / 771832 | 57.0 / 69.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-deep | 0.0% | – | – / – | 145941 / 1190177 | 57.0 / 69.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-keywords | 0.0% | – | – / – | 175 / 4463 | 57.0 / 69.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-lite | 0.0% | – | – / – | 36 / 97 | 57.0 / 69.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-neighbors | 0.0% | – | – / – | 66 / 137 | 57.0 / 69.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-shape | 0.0% | – | – / – | 49 / 151 | 57.0 / 69.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-blend | 0.0% | – | – / – | 618 / 2399 | 57.0 / 69.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-memo | 0.0% | – | – / – | 96 / 438 | 57.0 / 69.0 | $0.000030 | $0.000000 |
| customer_frustration | understudent-auto | 0.0% | – | – / – | 205 / 4374 | 57.0 / 69.0 | $0.000030 | $0.000000 |
| incident_severity | hunch | 0.0% | – | – / – | 154616 / 2317135 | 59.0 / 121.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-probe | 0.0% | – | – / – | 191645 / 1472724 | 59.0 / 121.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-deep | 0.0% | – | – / – | 236602 / 1840365 | 59.0 / 121.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-keywords | 0.0% | – | – / – | 239 / 2367 | 59.0 / 121.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-lite | 0.0% | – | – / – | 37 / 122 | 59.0 / 121.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-neighbors | 0.0% | – | – / – | 50 / 111 | 59.0 / 121.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-shape | 0.0% | – | – / – | 85 / 176 | 59.0 / 121.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-blend | 0.0% | – | – / – | 821 / 2225 | 59.0 / 121.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-memo | 0.0% | – | – / – | 253 / 504 | 59.0 / 121.0 | $0.000034 | $0.000000 |
| incident_severity | understudent-auto | 0.0% | – | – / – | 280 / 796 | 59.0 / 121.0 | $0.000034 | $0.000000 |
| output_kind | hunch | 57.0% | 98.0% (95.8%) n=342 | 92936 / 603480 | 79598 / 391388 | 57.0 / 83.0 | $0.000032 | $0.017974 |
| output_kind | understudent-probe | 57.0% | 98.0% (95.8%) n=342 | 188682 / 1808761 | 59901 / 1598464 | 57.0 / 83.0 | $0.000032 | $0.017974 |
| output_kind | understudent-deep | 63.7% | 97.9% (95.9%) n=382 | 116225 / 901668 | 129871 / 1096677 | 57.0 / 83.0 | $0.000032 | $0.020076 |
| output_kind | understudent-keywords | 51.0% | 98.0% (95.8%) n=306 | 104 / 516 | 73 / 500 | 57.0 / 83.0 | $0.000032 | $0.016082 |
| output_kind | understudent-lite | 52.2% | 95.2% (92.2%) n=313 | 166 / 1278 | 223 / 5190 | 57.0 / 83.0 | $0.000032 | $0.016450 |
| output_kind | understudent-neighbors | 43.8% | 97.7% (95.1%) n=263 | 58 / 91 | 36 / 76 | 57.0 / 83.0 | $0.000032 | $0.013822 |
| output_kind | understudent-shape | 0.0% | – | – / – | 60 / 236 | 57.0 / 83.0 | $0.000032 | $0.000000 |
| output_kind | understudent-blend | 50.8% | 98.7% (96.7%) n=305 | 433 / 2895 | 341 / 3134 | 57.0 / 83.0 | $0.000032 | $0.016029 |
| output_kind | understudent-memo | 0.0% | – | – / – | 69 / 426 | 57.0 / 83.0 | $0.000032 | $0.000000 |
| output_kind | understudent-auto | 63.7% | 97.9% (95.9%) n=382 | 91 / 1000 | 153 / 1359 | 57.0 / 83.0 | $0.000032 | $0.020076 |
| passage_answers | hunch | 0.0% | – | – / – | 104244 / 791268 | 57.0 / 739.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-probe | 0.0% | – | – / – | 356952 / 1166185 | 57.0 / 739.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-deep | 15.7% | 94.7% (88.1%) n=94 | 566012 / 1494721 | 532786 / 1636362 | 57.0 / 739.0 | $0.000034 | $0.005355 |
| passage_answers | understudent-keywords | 0.0% | – | – / – | 250 / 4547 | 57.0 / 739.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-lite | 0.0% | – | – / – | 920 / 6767 | 57.0 / 739.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-neighbors | 0.0% | – | – / – | 97 / 2695 | 57.0 / 739.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-shape | 0.0% | – | – / – | 118 / 2363 | 57.0 / 739.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-blend | 0.0% | – | – / – | 1370 / 12154 | 57.0 / 739.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-memo | 0.0% | – | – / – | 167 / 2283 | 57.0 / 739.0 | $0.000034 | $0.000000 |
| passage_answers | understudent-auto | 15.7% | 94.7% (88.1%) n=94 | 153 / 393 | 335 / 4569 | 57.0 / 739.0 | $0.000034 | $0.005355 |
| payment_flip | hunch | 47.2% | 89.4% (85.3%) n=283 | 238004 / 795509 | 238761 / 911535 | 57.0 / 112.0 | $0.000027 | $0.012825 |
| payment_flip | understudent-probe | 47.2% | 89.4% (85.3%) n=283 | 101928 / 608699 | 96210 / 569983 | 57.0 / 112.0 | $0.000027 | $0.012825 |
| payment_flip | understudent-deep | 69.7% | 92.3% (89.4%) n=418 | 496255 / 1072595 | 340462 / 1087578 | 57.0 / 112.0 | $0.000027 | $0.018943 |
| payment_flip | understudent-keywords | 13.8% | 98.8% (93.5%) n=83 | 115 / 5623 | 69 / 643 | 57.0 / 112.0 | $0.000027 | $0.003761 |
| payment_flip | understudent-lite | 24.8% | 100.0% (97.5%) n=149 | 61 / 2945 | 26 / 117 | 57.0 / 112.0 | $0.000027 | $0.006752 |
| payment_flip | understudent-neighbors | 0.0% | – | – / – | 64 / 286 | 57.0 / 112.0 | $0.000027 | $0.000000 |
| payment_flip | understudent-shape | 0.0% | – | – / – | 64 / 3894 | 57.0 / 112.0 | $0.000027 | $0.000000 |
| payment_flip | understudent-blend | 19.7% | 100.0% (96.8%) n=118 | 182 / 559 | 277 / 6846 | 57.0 / 112.0 | $0.000027 | $0.005347 |
| payment_flip | understudent-memo | 0.0% | – | – / – | 58 / 154 | 57.0 / 112.0 | $0.000027 | $0.000000 |
| payment_flip | understudent-auto | 69.7% | 92.3% (89.4%) n=418 | 87 / 4165 | 146 / 57524 | 57.0 / 112.0 | $0.000027 | $0.018943 |
| politer_reply | hunch | 0.0% | – | – / – | 732324 / 2155113 | 57.0 / 186.0 | $0.000033 | $0.000000 |
| politer_reply | understudent-probe | 78.2% | 99.8% (98.8%) n=469 | 185829 / 1001290 | 567943 / 1197072 | 57.0 / 186.0 | $0.000033 | $0.026004 |
| politer_reply | understudent-deep | 78.2% | 98.3% (96.7%) n=469 | 417344 / 1898935 | 256612 / 1166528 | 57.0 / 186.0 | $0.000033 | $0.026004 |
| politer_reply | understudent-keywords | 80.0% | 94.2% (91.7%) n=480 | 174 / 615 | 124 / 383 | 57.0 / 186.0 | $0.000033 | $0.026614 |
| politer_reply | understudent-lite | 73.0% | 97.3% (95.3%) n=438 | 92 / 7457 | 473 / 23896 | 57.0 / 186.0 | $0.000033 | $0.024285 |
| politer_reply | understudent-neighbors | 63.8% | 95.8% (93.3%) n=383 | 125 / 2257 | 51 / 629 | 57.0 / 186.0 | $0.000033 | $0.021235 |
| politer_reply | understudent-shape | 58.5% | 90.3% (86.8%) n=351 | 113 / 383 | 88 / 567 | 57.0 / 186.0 | $0.000033 | $0.019461 |
| politer_reply | understudent-blend | 76.3% | 96.3% (94.1%) n=458 | 942 / 12054 | 777 / 14741 | 57.0 / 186.0 | $0.000033 | $0.025394 |
| politer_reply | understudent-memo | 0.0% | – | – / – | 159 / 582 | 57.0 / 186.0 | $0.000033 | $0.000000 |
| politer_reply | understudent-auto | 80.0% | 94.2% (91.7%) n=480 | 266 / 551 | 436 / 6595 | 57.0 / 186.0 | $0.000033 | $0.026614 |
| retry_exception | hunch | 52.8% | 93.4% (90.1%) n=317 | 939957 / 3677206 | 671543 / 3608446 | 59.0 / 123.0 | $0.000044 | $0.023181 |
| retry_exception | understudent-probe | 50.8% | 94.4% (91.3%) n=305 | 938957 / 3665136 | 1132017 / 4043881 | 59.0 / 123.0 | $0.000044 | $0.022304 |
| retry_exception | understudent-deep | 51.7% | 97.1% (94.6%) n=310 | 1102125 / 5521681 | 1137177 / 5338208 | 59.0 / 123.0 | $0.000044 | $0.022669 |
| retry_exception | understudent-keywords | 26.7% | 98.8% (95.6%) n=160 | 480 / 14575 | 448 / 21917 | 59.0 / 123.0 | $0.000044 | $0.011700 |
| retry_exception | understudent-lite | 12.7% | 94.7% (87.2%) n=76 | 911 / 15088 | 1922 / 48152 | 59.0 / 123.0 | $0.000044 | $0.005558 |
| retry_exception | understudent-neighbors | 0.0% | – | – / – | 69 / 318 | 59.0 / 123.0 | $0.000044 | $0.000000 |
| retry_exception | understudent-shape | 0.0% | – | – / – | 205 / 637 | 59.0 / 123.0 | $0.000044 | $0.000000 |
| retry_exception | understudent-blend | 15.0% | 95.6% (89.1%) n=90 | 3834 / 45588 | 2314 / 40624 | 59.0 / 123.0 | $0.000044 | $0.006581 |
| retry_exception | understudent-memo | 0.0% | – | – / – | 301 / 8534 | 59.0 / 123.0 | $0.000044 | $0.000000 |
| retry_exception | understudent-auto | 51.3% | 94.8% (91.7%) n=308 | 67 / 8707 | 873 / 22164 | 59.0 / 123.0 | $0.000044 | $0.022523 |
| secret_needle | hunch | 0.0% | – | – / – | 844141 / 3511691 | 66.0 / 90.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-probe | 0.0% | – | – / – | 945463 / 3238239 | 66.0 / 90.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-deep | 0.0% | – | – / – | 1701873 / 9057513 | 66.0 / 90.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-keywords | 0.0% | – | – / – | 1054 / 9984 | 66.0 / 90.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-lite | 0.0% | – | – / – | 1963 / 9229 | 66.0 / 90.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-neighbors | 0.0% | – | – / – | 62 / 208 | 66.0 / 90.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-shape | 0.0% | – | – / – | 383 / 4657 | 66.0 / 90.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-blend | 0.0% | – | – / – | 4336 / 27506 | 66.0 / 90.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-memo | 0.0% | – | – / – | 467 / 5307 | 66.0 / 90.0 | $0.000072 | $0.000000 |
| secret_needle | understudent-auto | 0.0% | – | – / – | 1214 / 9364 | 66.0 / 90.0 | $0.000072 | $0.000000 |
| tool_failed | hunch | 39.2% | 97.4% (94.5%) n=235 | 644377 / 2468990 | 121308 / 1623570 | 86.0 / 183.0 | $0.000031 | $0.012164 |
| tool_failed | understudent-probe | 39.2% | 97.4% (94.5%) n=235 | 47 / 109 | 28 / 133 | 86.0 / 183.0 | $0.000031 | $0.012164 |
| tool_failed | understudent-deep | 28.2% | 97.6% (94.1%) n=169 | 104 / 8576 | 92 / 8169 | 86.0 / 183.0 | $0.000031 | $0.008747 |
| tool_failed | understudent-keywords | 32.3% | 98.5% (95.6%) n=194 | 142 / 4309 | 115 / 4333 | 86.0 / 183.0 | $0.000031 | $0.010041 |
| tool_failed | understudent-lite | 0.0% | – | – / – | 30 / 97 | 86.0 / 183.0 | $0.000031 | $0.000000 |
| tool_failed | understudent-neighbors | 0.0% | – | – / – | 61 / 2124 | 86.0 / 183.0 | $0.000031 | $0.000000 |
| tool_failed | understudent-shape | 0.0% | – | – / – | 84 / 4086 | 86.0 / 183.0 | $0.000031 | $0.000000 |
| tool_failed | understudent-blend | 0.0% | – | – / – | 562 / 7776 | 86.0 / 183.0 | $0.000031 | $0.000000 |
| tool_failed | understudent-memo | 0.0% | – | – / – | 91 / 4127 | 86.0 / 183.0 | $0.000031 | $0.000000 |
| tool_failed | understudent-auto | 39.2% | 97.4% (94.5%) n=235 | 48 / 123 | 160 / 8415 | 86.0 / 183.0 | $0.000031 | $0.012164 |

## Timing

| dataset | system | wall s | train s | items | status |
|---|---|---|---|---|---|
| capital_city | hunch | 24.6 | – | 600 | ok |
| capital_city | understudent-probe | 12.3 | – | 600 | ok |
| capital_city | understudent-deep | 21.2 | – | 600 | ok |
| capital_city | understudent-keywords | 0.1 | – | 600 | ok |
| capital_city | understudent-lite | 1.2 | – | 600 | ok |
| capital_city | understudent-neighbors | 1.2 | – | 600 | ok |
| capital_city | understudent-shape | 0.1 | – | 600 | ok |
| capital_city | understudent-blend | 1.6 | – | 600 | ok |
| capital_city | understudent-memo | 0.1 | – | 600 | ok |
| capital_city | understudent-auto | 5.0 | – | 600 | ok |
| code_lookup | hunch | 42.8 | – | 600 | ok |
| code_lookup | understudent-probe | 31.1 | – | 600 | ok |
| code_lookup | understudent-deep | 90.2 | – | 600 | ok |
| code_lookup | understudent-keywords | 1.4 | – | 600 | ok |
| code_lookup | understudent-lite | 5.0 | – | 600 | ok |
| code_lookup | understudent-neighbors | 5.0 | – | 600 | ok |
| code_lookup | understudent-shape | 0.5 | – | 600 | ok |
| code_lookup | understudent-blend | 7.7 | – | 600 | ok |
| code_lookup | understudent-memo | 2.1 | – | 600 | ok |
| code_lookup | understudent-auto | 30.2 | – | 600 | ok |
| customer_frustration | hunch | 115.1 | – | 600 | ok |
| customer_frustration | understudent-probe | 74.0 | – | 600 | ok |
| customer_frustration | understudent-deep | 165.1 | – | 600 | ok |
| customer_frustration | understudent-keywords | 3.6 | – | 600 | ok |
| customer_frustration | understudent-lite | 1.9 | – | 600 | ok |
| customer_frustration | understudent-neighbors | 1.6 | – | 600 | ok |
| customer_frustration | understudent-shape | 0.2 | – | 600 | ok |
| customer_frustration | understudent-blend | 6.0 | – | 600 | ok |
| customer_frustration | understudent-memo | 0.2 | – | 600 | ok |
| customer_frustration | understudent-auto | 29.1 | – | 600 | ok |
| incident_severity | hunch | 259.8 | – | 600 | ok |
| incident_severity | understudent-probe | 137.5 | – | 600 | ok |
| incident_severity | understudent-deep | 259.0 | – | 600 | ok |
| incident_severity | understudent-keywords | 3.6 | – | 600 | ok |
| incident_severity | understudent-lite | 1.5 | – | 600 | ok |
| incident_severity | understudent-neighbors | 1.5 | – | 600 | ok |
| incident_severity | understudent-shape | 0.2 | – | 600 | ok |
| incident_severity | understudent-blend | 8.1 | – | 600 | ok |
| incident_severity | understudent-memo | 0.5 | – | 600 | ok |
| incident_severity | understudent-auto | 35.1 | – | 600 | ok |
| output_kind | hunch | 79.0 | – | 600 | ok |
| output_kind | understudent-probe | 162.0 | – | 600 | ok |
| output_kind | understudent-deep | 125.3 | – | 600 | ok |
| output_kind | understudent-keywords | 1.3 | – | 600 | ok |
| output_kind | understudent-lite | 1.5 | – | 600 | ok |
| output_kind | understudent-neighbors | 1.2 | – | 600 | ok |
| output_kind | understudent-shape | 0.2 | – | 600 | ok |
| output_kind | understudent-blend | 2.4 | – | 600 | ok |
| output_kind | understudent-memo | 0.1 | – | 600 | ok |
| output_kind | understudent-auto | 5.9 | – | 600 | ok |
| passage_answers | hunch | 98.3 | – | 600 | ok |
| passage_answers | understudent-probe | 240.3 | – | 600 | ok |
| passage_answers | understudent-deep | 396.9 | – | 600 | ok |
| passage_answers | understudent-keywords | 2.4 | – | 600 | ok |
| passage_answers | understudent-lite | 4.0 | – | 600 | ok |
| passage_answers | understudent-neighbors | 3.2 | – | 600 | ok |
| passage_answers | understudent-shape | 0.3 | – | 600 | ok |
| passage_answers | understudent-blend | 6.3 | – | 600 | ok |
| passage_answers | understudent-memo | 0.7 | – | 600 | ok |
| passage_answers | understudent-auto | 43.7 | – | 600 | ok |
| payment_flip | hunch | 180.6 | – | 600 | ok |
| payment_flip | understudent-probe | 92.0 | – | 600 | ok |
| payment_flip | understudent-deep | 272.4 | – | 600 | ok |
| payment_flip | understudent-keywords | 0.6 | – | 600 | ok |
| payment_flip | understudent-lite | 4.0 | – | 600 | ok |
| payment_flip | understudent-neighbors | 3.8 | – | 600 | ok |
| payment_flip | understudent-shape | 0.6 | – | 600 | ok |
| payment_flip | understudent-blend | 4.6 | – | 600 | ok |
| payment_flip | understudent-memo | 0.5 | – | 600 | ok |
| payment_flip | understudent-auto | 9.5 | – | 600 | ok |
| politer_reply | hunch | 514.3 | – | 600 | ok |
| politer_reply | understudent-probe | 208.2 | – | 600 | ok |
| politer_reply | understudent-deep | 313.6 | – | 600 | ok |
| politer_reply | understudent-keywords | 0.2 | – | 600 | ok |
| politer_reply | understudent-lite | 2.9 | – | 600 | ok |
| politer_reply | understudent-neighbors | 2.5 | – | 600 | ok |
| politer_reply | understudent-shape | 0.1 | – | 600 | ok |
| politer_reply | understudent-blend | 3.4 | – | 600 | ok |
| politer_reply | understudent-memo | 0.3 | – | 600 | ok |
| politer_reply | understudent-auto | 4.9 | – | 600 | ok |
| retry_exception | hunch | 600.6 | – | 600 | ok |
| retry_exception | understudent-probe | 734.4 | – | 600 | ok |
| retry_exception | understudent-deep | 993.7 | – | 600 | ok |
| retry_exception | understudent-keywords | 2.2 | – | 600 | ok |
| retry_exception | understudent-lite | 10.8 | – | 600 | ok |
| retry_exception | understudent-neighbors | 7.6 | – | 600 | ok |
| retry_exception | understudent-shape | 0.6 | – | 600 | ok |
| retry_exception | understudent-blend | 12.0 | – | 600 | ok |
| retry_exception | understudent-memo | 1.8 | – | 600 | ok |
| retry_exception | understudent-auto | 81.5 | – | 600 | ok |
| secret_needle | hunch | 587.4 | – | 600 | ok |
| secret_needle | understudent-probe | 678.4 | – | 600 | ok |
| secret_needle | understudent-deep | 1548.2 | – | 600 | ok |
| secret_needle | understudent-keywords | 5.9 | – | 600 | ok |
| secret_needle | understudent-lite | 4.4 | – | 600 | ok |
| secret_needle | understudent-neighbors | 3.1 | – | 600 | ok |
| secret_needle | understudent-shape | 0.8 | – | 600 | ok |
| secret_needle | understudent-blend | 13.7 | – | 600 | ok |
| secret_needle | understudent-memo | 1.4 | – | 600 | ok |
| secret_needle | understudent-auto | 141.4 | – | 600 | ok |
| tool_failed | hunch | 278.5 | – | 600 | ok |
| tool_failed | understudent-probe | 4.0 | – | 600 | ok |
| tool_failed | understudent-deep | 27.6 | – | 600 | ok |
| tool_failed | understudent-keywords | 0.7 | – | 600 | ok |
| tool_failed | understudent-lite | 5.1 | – | 600 | ok |
| tool_failed | understudent-neighbors | 4.1 | – | 600 | ok |
| tool_failed | understudent-shape | 0.4 | – | 600 | ok |
| tool_failed | understudent-blend | 7.2 | – | 600 | ok |
| tool_failed | understudent-memo | 1.1 | – | 600 | ok |
| tool_failed | understudent-auto | 23.4 | – | 600 | ok |

Notes: `audit agree` covers local answers the lead audited and forwarded items the understudent would have answered (eligible). hunch fields that are approximations are listed in the docstring of bench/hunch_runner/run_hunch.py.
