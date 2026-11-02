# understudy-bench run `v0`


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
| world_knowledge | capital_city | noul | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.3% | 73.0% | 0/600 | 22795 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 72.8% | 0/600 | 21287 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 73.4% | 0/600 | 30146 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 69.8% | 0/600 | 36 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 57.4% | 0/600 | 88 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 72.0% | 0/600 | 17747 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-ladder | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 73.4% | 0/600 | 26303 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | hunch-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.8% | 73.0% | 0/400 | 19851 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-probe-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 72.8% | 0/400 | 17513 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-deep-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 73.0% | 0/400 | 28177 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-keywords-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 69.8% | 0/400 | 42 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-lite-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 55.0% | 0/400 | 76 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-neighbors-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 71.8% | 0/400 | 17792 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-ladder-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 73.0% | 0/400 | 26740 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 45.0% | 0/600 | 50617 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 45.0% | 0/600 | 44970 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 56.4% | 0/600 | 69247 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 25.0% | 0/600 | 69 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 26.4% | 0/600 | 171 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 34.6% | 0/600 | 52377 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-ladder | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 56.4% | 0/600 | 80568 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | hunch-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 44.5% | 0/400 | 66434 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-probe-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 44.5% | 0/400 | 46016 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-deep-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 59.5% | 0/400 | 72571 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-keywords-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 24.5% | 0/400 | 83 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-lite-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 26.0% | 0/400 | 154 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-neighbors-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 33.5% | 0/400 | 43098 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-ladder-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 59.5% | 0/400 | 66190 | **correctly abstains** | abstains |
| graded_tone | customer_frustration | score | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 54.4% | 0/600 | 82265 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 54.4% | 0/600 | 66876 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 56.8% | 0/600 | 91822 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 49.8% | 0/600 | 193 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 50.8% | 0/600 | 348 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 47.6% | 0/600 | 63457 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-ladder | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 56.8% | 0/600 | 93542 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | hunch-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 56.0% | 0/400 | 79665 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-probe-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 56.0% | 0/400 | 61922 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-deep-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 58.5% | 0/400 | 91920 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-keywords-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 53.0% | 0/400 | 187 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-lite-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 54.0% | 0/400 | 369 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-neighbors-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 50.0% | 0/400 | 71985 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-ladder-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 58.5% | 0/400 | 116760 | **correctly abstains** | partial |
| compound_judgment | incident_severity | score | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 51.6% | 0/600 | 147621 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 51.6% | 0/600 | 115702 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 54.4% | 0/600 | 181281 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 50.2% | 0/600 | 299 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 48.8% | 0/600 | 509 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 47.2% | 0/600 | 132002 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-ladder | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 54.4% | 0/600 | 154981 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | hunch-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.0% | 67.0% | 52.2% | 0/400 | 110397 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-probe-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.0% | 67.0% | 52.2% | 0/400 | 107254 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-deep-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.0% | 67.0% | 55.8% | 0/400 | 169569 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-keywords-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.0% | 67.0% | 52.0% | 0/400 | 305 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-lite-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.0% | 67.0% | 51.5% | 0/400 | 467 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-neighbors-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.0% | 67.0% | 49.0% | 0/400 | 109097 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-ladder-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.0% | 67.0% | 55.8% | 0/400 | 170362 | **correctly abstains** | abstains |
| topic_kind | output_kind | choice | hunch | yes | 101 | 57.0% | 79.0% | 98.0% (342) | 100.0% (34) | 25.7% | 0.97 | 97.6% (254) | 95.8% | 95.0% | 90.2% | 308/600 | 110205 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-probe | yes | 101 | 57.0% | 79.0% | 98.0% (342) | 100.0% (34) | 25.7% | 0.97 | 97.6% (254) | 95.8% | 95.0% | 90.2% | 308/600 | 90584 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-deep | yes | 101 | 63.7% | 80.7% | 97.9% (382) | 100.0% (40) | 24.1% | 0.97 | 97.6% (290) | 95.8% | 95.5% | 93.0% | 342/600 | 145792 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-keywords | yes | 102 | 51.0% | 69.7% | 98.0% (306) | 96.4% (28) | 29.4% | 0.98 | 97.2% (216) | 95.8% | 96.0% | 87.6% | 278/600 | 186 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-lite | yes | 101 | 52.2% | 65.7% | 95.2% (313) | 100.0% (32) | 26.2% | 0.94 | 94.4% (231) | 95.8% | 95.0% | 87.6% | 281/600 | 315 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-neighbors | yes | 201 | 43.8% | 68.3% | 97.7% (263) | 100.0% (23) | 27.0% | 0.97 | 96.9% (192) | 95.8% | 94.8% | 83.8% | 240/600 | 99891 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-ladder | yes | 101 | 64.8% | 80.3% | 97.2% (389) | 87.5% (40) | 24.4% | 0.96 | 96.6% (294) | 95.8% | 95.7% | 93.6% | 349/600 | 98184 | **takes over** | takes over |
| topic_kind | output_kind | choice | hunch-split | yes | 1 | 83.5% | 84.0% | 96.4% (334) | 94.1% (34) | 24.9% | 0.95 | 95.6% (251) | 96.5% | 95.0% | 91.2% | 300/400 | 102595 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-probe-split | yes | 1 | 83.5% | 84.0% | 96.4% (334) | 94.1% (34) | 24.9% | 0.95 | 95.6% (251) | 96.5% | 95.0% | 91.2% | 300/400 | 96297 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-deep-split | yes | 1 | 91.0% | 90.5% | 95.6% (364) | 94.3% (35) | 23.6% | 0.94 | 94.6% (278) | 96.5% | 95.2% | 93.8% | 329/400 | 141578 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-keywords-split | yes | 1 | 75.5% | 75.0% | 97.4% (302) | 92.9% (28) | 25.8% | 0.97 | 96.4% (224) | 96.5% | 96.8% | 91.8% | 274/400 | 177 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-lite-split | yes | 1 | 81.5% | 81.0% | 93.6% (326) | 97.0% (33) | 23.3% | 0.92 | 92.4% (250) | 96.5% | 94.0% | 89.2% | 293/400 | 432 | **harmful** | takes over |
| topic_kind | output_kind | choice | understudent-neighbors-split | yes | 1 | 65.8% | 72.0% | 97.7% (263) | 100.0% (23) | 27.0% | 0.97 | 96.9% (192) | 96.5% | 95.0% | 87.8% | 240/400 | 112875 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-ladder-split | yes | 1 | 94.0% | 93.5% | 95.5% (376) | 100.0% (40) | 23.4% | 0.94 | 94.4% (288) | 96.5% | 96.0% | 94.2% | 336/400 | 42691 | **takes over** | takes over |
| relational_match | passage_answers | noul | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.8% | 75.0% | 0/600 | 107403 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 80.2% | 0/600 | 78450 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-deep | yes | 301 | 15.7% | 31.3% | 94.7% (94) | 100.0% (4) | 77.7% | 0.84 | 85.7% (21) | 93.7% | 93.2% | 81.8% | 90/600 | 101319 | **partial** | abstains |
| relational_match | passage_answers | noul | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 75.4% | 0/600 | 220 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 75.8% | 0/600 | 599 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 67.4% | 0/600 | 85668 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-ladder | yes | 301 | 15.7% | 31.3% | 94.7% (94) | 100.0% (4) | 77.7% | 0.84 | 85.7% (21) | 93.7% | 93.2% | 81.8% | 90/600 | 112979 | **partial** | abstains |
| relational_match | passage_answers | noul | hunch-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 94.0% | 94.2% | 73.8% | 0/400 | 83904 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-probe-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 94.0% | 94.0% | 79.8% | 0/400 | 68067 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-deep-split | yes | 101 | 23.5% | 23.0% | 94.7% (94) | 100.0% (4) | 77.7% | 0.84 | 85.7% (21) | 94.0% | 93.2% | 80.5% | 90/400 | 98723 | **partial** | abstains |
| relational_match | passage_answers | noul | understudent-keywords-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 94.0% | 94.0% | 74.0% | 0/400 | 320 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-lite-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 94.0% | 94.0% | 75.0% | 0/400 | 734 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-neighbors-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 94.0% | 94.0% | 66.0% | 0/400 | 83322 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-ladder-split | yes | 101 | 23.5% | 23.0% | 94.7% (94) | 100.0% (4) | 77.7% | 0.84 | 85.7% (21) | 94.0% | 93.2% | 80.5% | 90/400 | 114914 | **partial** | abstains |
| minimal_pair | payment_flip | noul | hunch | yes | 102 | 47.2% | 50.0% | 89.4% (283) | 84.6% (26) | 68.9% | 0.76 | 86.4% (88) | 99.7% | 95.0% | 81.6% | 257/600 | 35254 | **harmful** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-probe | yes | 102 | 47.2% | 50.0% | 89.4% (283) | 84.6% (26) | 68.9% | 0.76 | 86.4% (88) | 99.7% | 95.0% | 81.6% | 257/600 | 41326 | **harmful** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-deep | yes | 101 | 69.7% | 87.0% | 92.3% (418) | 90.9% (44) | 69.6% | 0.82 | 85.0% (127) | 99.7% | 94.7% | 87.0% | 374/600 | 59113 | **harmful** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-keywords | yes | 501 | 13.8% | 27.7% | 98.8% (83) | 100.0% (4) | 72.3% | 0.97 | 95.7% (23) | 99.7% | 99.5% | 92.0% | 79/600 | 64 | **partial** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-lite | yes | 303 | 24.8% | 49.7% | 100.0% (149) | 100.0% (11) | 71.8% | 1.00 | 100.0% (42) | 99.7% | 99.7% | 85.2% | 138/600 | 175 | **partial** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 72.2% | 0/600 | 38238 | **correctly abstains** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-ladder | yes | 101 | 69.7% | 87.0% | 92.3% (418) | 90.9% (44) | 69.6% | 0.82 | 85.0% (127) | 99.7% | 94.7% | 87.0% | 374/600 | 59776 | **harmful** | partial or abstains |
| minimal_pair | payment_flip | noul | hunch-split | yes | 1 | 73.0% | 80.0% | 96.6% (292) | 100.0% (28) | 71.9% | 0.91 | 92.7% (82) | 100.0% | 97.5% | 88.5% | 264/400 | 50516 | **harmful** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-probe-split | yes | 1 | 73.0% | 80.0% | 96.6% (292) | 100.0% (28) | 71.9% | 0.91 | 92.7% (82) | 100.0% | 97.5% | 88.2% | 264/400 | 41258 | **harmful** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-deep-split | yes | 1 | 88.8% | 90.0% | 96.9% (355) | 97.1% (34) | 72.4% | 0.92 | 92.9% (98) | 100.0% | 97.2% | 94.2% | 321/400 | 60794 | **harmful** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-keywords-split | yes | 301 | 20.8% | 41.5% | 98.8% (83) | 100.0% (4) | 72.3% | 0.97 | 95.7% (23) | 100.0% | 99.8% | 94.2% | 79/400 | 82 | **partial** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-lite-split | yes | 103 | 37.2% | 58.5% | 100.0% (149) | 100.0% (11) | 71.8% | 1.00 | 100.0% (42) | 100.0% | 100.0% | 89.8% | 138/400 | 150 | **takes over** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-neighbors-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 73.5% | 0/400 | 36099 | **correctly abstains** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-ladder-split | yes | 1 | 88.8% | 90.0% | 96.9% (355) | 97.1% (34) | 72.4% | 0.92 | 92.9% (98) | 100.0% | 97.2% | 94.2% | 321/400 | 62051 | **harmful** | partial or abstains |
| pair_compare | politer_reply | choice | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.8% | 99.8% | 80.0% | 0/600 | 101944 | **correctly abstains** | abstains |
| pair_compare | politer_reply | choice | understudent-probe | yes | 101 | 78.2% | 95.3% | 99.8% (469) | 100.0% (49) | 51.6% | 1.00 | 99.6% (227) | 99.8% | 99.7% | 99.0% | 420/600 | 77534 | **takes over** | ? |
| pair_compare | politer_reply | choice | understudent-deep | yes | 101 | 78.2% | 95.3% | 98.3% (469) | 100.0% (49) | 51.6% | 0.97 | 99.6% (227) | 99.8% | 98.5% | 97.6% | 420/600 | 113562 | **partial** | ? |
| pair_compare | politer_reply | choice | understudent-keywords | yes | 101 | 80.0% | 97.7% | 94.2% (480) | 90.0% (50) | 52.5% | 0.88 | 97.8% (228) | 99.8% | 95.2% | 93.6% | 430/600 | 229 | **harmful** | ? |
| pair_compare | politer_reply | choice | understudent-lite | yes | 101 | 73.0% | 90.0% | 97.3% (438) | 95.7% (47) | 50.9% | 0.95 | 98.6% (215) | 99.8% | 97.8% | 96.0% | 391/600 | 576 | **partial** | ? |
| pair_compare | politer_reply | choice | understudent-neighbors | yes | 201 | 63.8% | 96.0% | 95.8% (383) | 92.5% (40) | 53.3% | 0.92 | 100.0% (179) | 99.8% | 97.2% | 95.4% | 343/600 | 81322 | **harmful** | ? |
| pair_compare | politer_reply | choice | understudent-ladder | yes | 101 | 82.7% | 99.3% | 94.4% (496) | 96.2% (53) | 51.8% | 0.89 | 97.9% (239) | 99.8% | 95.2% | 94.0% | 443/600 | 20369 | **harmful** | ? |
| pair_compare | politer_reply | choice | hunch-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.8% | 99.8% | 80.2% | 0/400 | 92863 | **correctly abstains** | abstains |
| pair_compare | politer_reply | choice | understudent-probe-split | yes | 1 | 95.8% | 96.0% | 99.5% (383) | 97.5% (40) | 53.3% | 0.99 | 100.0% (179) | 99.8% | 99.2% | 98.8% | 343/400 | 73092 | **takes over** | ? |
| pair_compare | politer_reply | choice | understudent-deep-split | yes | 1 | 95.8% | 96.0% | 99.5% (383) | 97.5% (40) | 53.3% | 0.99 | 100.0% (179) | 99.8% | 99.2% | 98.8% | 343/400 | 103602 | **takes over** | ? |
| pair_compare | politer_reply | choice | understudent-keywords-split | yes | 1 | 97.5% | 97.5% | 96.2% (390) | 95.0% (40) | 53.6% | 0.92 | 98.3% (181) | 99.8% | 96.0% | 95.5% | 350/400 | 187 | **harmful** | ? |
| pair_compare | politer_reply | choice | understudent-lite-split | yes | 1 | 93.0% | 95.0% | 98.9% (372) | 97.4% (38) | 53.2% | 0.98 | 98.9% (174) | 99.8% | 98.8% | 97.2% | 334/400 | 293 | **takes over** | ? |
| pair_compare | politer_reply | choice | understudent-neighbors-split | yes | 1 | 95.8% | 96.0% | 95.8% (383) | 92.5% (40) | 53.3% | 0.92 | 100.0% (179) | 99.8% | 95.8% | 94.5% | 343/400 | 70036 | **harmful** | ? |
| pair_compare | politer_reply | choice | understudent-ladder-split | yes | 1 | 99.8% | 100.0% | 96.2% (399) | 97.5% (40) | 53.4% | 0.92 | 98.4% (186) | 99.8% | 96.0% | 95.8% | 359/400 | 2709 | **harmful** | ? |
| conditional_rule | retry_exception | noul | hunch | yes | 201 | 52.8% | 77.0% | 93.4% (317) | 96.9% (32) | 76.0% | 0.81 | 76.3% (76) | 100.0% | 96.5% | 88.8% | 285/600 | 327983 | **harmful** | abstains |
| conditional_rule | retry_exception | noul | understudent-probe | yes | 204 | 50.8% | 75.7% | 94.4% (305) | 96.4% (28) | 76.7% | 0.83 | 77.5% (71) | 100.0% | 97.2% | 89.4% | 277/600 | 319543 | **harmful** | abstains |
| conditional_rule | retry_exception | noul | understudent-deep | yes | 201 | 49.7% | 76.3% | 97.7% (298) | 100.0% (28) | 79.2% | 0.93 | 88.7% (62) | 100.0% | 98.8% | 91.2% | 270/600 | 536489 | **partial** | abstains |
| conditional_rule | retry_exception | noul | understudent-keywords | yes | 401 | 26.7% | 53.3% | 98.8% (160) | 100.0% (12) | 80.6% | 0.96 | 93.5% (31) | 100.0% | 99.7% | 88.6% | 148/600 | 379 | **takes over** | abstains |
| conditional_rule | retry_exception | noul | understudent-lite | yes | 501 | 11.8% | 23.7% | 95.8% (71) | 100.0% (3) | 81.7% | 0.84 | 76.9% (13) | 100.0% | 99.5% | 83.6% | 68/600 | 798 | **partial** | abstains |
| conditional_rule | retry_exception | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 74.0% | 0/600 | 414751 | **correctly abstains** | abstains |
| conditional_rule | retry_exception | noul | understudent-ladder | yes | 201 | 51.2% | 79.3% | 97.7% (307) | 100.0% (29) | 79.2% | 0.93 | 89.1% (64) | 100.0% | 98.8% | 91.4% | 278/600 | 437929 | **partial** | abstains |
| conditional_rule | retry_exception | noul | hunch-split | yes | 1 | 79.2% | 79.0% | 93.4% (317) | 96.9% (32) | 76.0% | 0.81 | 76.3% (76) | 100.0% | 94.8% | 88.0% | 285/400 | 348282 | **harmful** | abstains |
| conditional_rule | retry_exception | noul | understudent-probe-split | yes | 4 | 76.2% | 79.5% | 94.4% (305) | 96.4% (28) | 76.7% | 0.83 | 77.5% (71) | 100.0% | 95.8% | 88.8% | 277/400 | 355170 | **harmful** | abstains |
| conditional_rule | retry_exception | noul | understudent-deep-split | yes | 1 | 74.5% | 82.5% | 97.7% (298) | 100.0% (28) | 79.2% | 0.93 | 88.7% (62) | 100.0% | 98.2% | 90.8% | 270/400 | 513315 | **partial** | abstains |
| conditional_rule | retry_exception | noul | understudent-keywords-split | yes | 201 | 40.0% | 80.0% | 98.8% (160) | 100.0% (12) | 80.6% | 0.96 | 93.5% (31) | 100.0% | 99.5% | 92.0% | 148/400 | 420 | **takes over** | abstains |
| conditional_rule | retry_exception | noul | understudent-lite-split | yes | 301 | 17.8% | 35.5% | 95.8% (71) | 100.0% (3) | 81.7% | 0.84 | 76.9% (13) | 100.0% | 99.2% | 86.5% | 68/400 | 840 | **partial** | abstains |
| conditional_rule | retry_exception | noul | understudent-neighbors-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 76.2% | 0/400 | 339704 | **correctly abstains** | abstains |
| conditional_rule | retry_exception | noul | understudent-ladder-split | yes | 1 | 76.8% | 87.0% | 97.7% (307) | 100.0% (29) | 79.2% | 0.93 | 89.1% (64) | 100.0% | 98.2% | 91.0% | 278/400 | 387582 | **partial** | abstains |
| needle | secret_needle | noul | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 68.0% | 0/600 | 609353 | **correctly abstains** | abstains or partial |
| needle | secret_needle | noul | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 67.8% | 0/600 | 607547 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 72.4% | 0/600 | 872800 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 71.0% | 0/600 | 766 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 64.4% | 0/600 | 1754 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 63.0% | 0/600 | 597119 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-ladder | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 72.4% | 0/600 | 1039690 | **correctly abstains** | ? |
| needle | secret_needle | noul | hunch-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 67.8% | 0/400 | 653273 | **correctly abstains** | abstains or partial |
| needle | secret_needle | noul | understudent-probe-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 67.5% | 0/400 | 583627 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-deep-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 72.5% | 0/400 | 764635 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-keywords-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 70.0% | 0/400 | 644 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-lite-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 64.5% | 0/400 | 1466 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-neighbors-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 61.3% | 0/400 | 554439 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-ladder-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 72.5% | 0/400 | 909776 | **correctly abstains** | ? |
| surface_property | tool_failed | noul | hunch | yes | 302 | 39.2% | 78.3% | 97.4% (235) | 94.7% (19) | 78.7% | 0.92 | 92.0% (50) | 92.3% | 92.7% | 90.0% | 216/600 | 126154 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-probe | yes | 302 | 39.2% | 78.3% | 97.4% (235) | 94.7% (19) | 78.7% | 0.92 | 92.0% (50) | 92.3% | 92.7% | 90.0% | 216/600 | 124214 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-deep | yes | 401 | 28.2% | 56.3% | 97.6% (169) | 100.0% (12) | 78.1% | 0.93 | 94.6% (37) | 92.3% | 92.7% | 89.6% | 157/600 | 168937 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-keywords | yes | 302 | 32.3% | 64.7% | 98.5% (194) | 100.0% (15) | 94.8% | 0.82 | 70.0% (10) | 92.3% | 92.8% | 81.8% | 179/600 | 279 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 92.3% | 92.3% | 82.6% | 0/600 | 389 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 92.3% | 92.3% | 83.4% | 0/600 | 109286 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-ladder | yes | 302 | 34.7% | 69.3% | 96.6% (208) | 93.8% (16) | 90.9% | 0.78 | 73.7% (19) | 92.3% | 92.8% | 86.6% | 192/600 | 108173 | **takes over** | takes over |
| surface_property | tool_failed | noul | hunch-split | yes | 102 | 58.8% | 78.0% | 97.4% (235) | 94.7% (19) | 78.7% | 0.92 | 92.0% (50) | 92.0% | 92.5% | 91.2% | 216/400 | 117507 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-probe-split | yes | 102 | 58.8% | 78.0% | 97.4% (235) | 94.7% (19) | 78.7% | 0.92 | 92.0% (50) | 92.0% | 92.5% | 91.2% | 216/400 | 106778 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-deep-split | yes | 201 | 42.2% | 84.5% | 97.6% (169) | 100.0% (12) | 78.1% | 0.93 | 94.6% (37) | 92.0% | 92.5% | 91.5% | 157/400 | 145080 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-keywords-split | yes | 102 | 48.5% | 66.0% | 98.5% (194) | 100.0% (15) | 94.8% | 0.82 | 70.0% (10) | 92.0% | 92.8% | 83.8% | 179/400 | 183 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-lite-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 92.0% | 92.0% | 84.5% | 0/400 | 351 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-neighbors-split | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 92.0% | 92.0% | 85.2% | 0/400 | 113797 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-ladder-split | yes | 102 | 52.0% | 73.0% | 96.6% (208) | 93.8% (16) | 90.9% | 0.78 | 73.7% (19) | 92.0% | 92.8% | 87.8% | 192/400 | 81561 | **takes over** | takes over |

## Winners by category (cold start)

Winner = a system that graduates with accuracy within 1 pt of the lead alone, ranked by 2nd-half coverage, ties to the lower µs/item.

| category | dataset | type | winner | cov. 2nd half | acc system | acc lead | µs/item | runner-up |
|---|---|---|---|---|---|---|---|---|
| world_knowledge | capital_city | noul | none | | | | | |
| binding_lookup | code_lookup | choice | none | | | | | |
| graded_tone | customer_frustration | score | none | | | | | |
| compound_judgment | incident_severity | score | none | | | | | |
| topic_kind | output_kind | choice | **understudent-deep** | 80.7% | 95.5% | 95.8% | 145792 | understudent-ladder (80.3%) |
| relational_match | passage_answers | noul | **understudent-deep** | 31.3% | 93.2% | 93.7% | 101319 | understudent-ladder (31.3%) |
| minimal_pair | payment_flip | noul | **understudent-lite** | 49.7% | 99.7% | 99.7% | 175 | understudent-keywords (27.7%) |
| pair_compare | politer_reply | choice | **understudent-probe** | 95.3% | 99.7% | 99.8% | 77534 | – |
| conditional_rule | retry_exception | noul | **understudent-keywords** | 53.3% | 99.7% | 100.0% | 379 | understudent-lite (23.7%) |
| needle | secret_needle | noul | none | | | | | |
| surface_property | tool_failed | noul | **understudent-probe** | 78.3% | 92.7% | 92.3% | 124214 | hunch (78.3%) |

## Winners by category (warm start, operate items)

Winner = a system that graduates with accuracy within 1 pt of the lead alone, ranked by 2nd-half coverage, ties to the lower µs/item.

| category | dataset | type | winner | cov. 2nd half | acc system | acc lead | µs/item | runner-up |
|---|---|---|---|---|---|---|---|---|
| world_knowledge | capital_city | noul | none | | | | | |
| binding_lookup | code_lookup | choice | none | | | | | |
| graded_tone | customer_frustration | score | none | | | | | |
| compound_judgment | incident_severity | score | none | | | | | |
| topic_kind | output_kind | choice | **understudent-ladder-split** | 93.5% | 96.0% | 96.5% | 42691 | understudent-keywords-split (75.0%) |
| relational_match | passage_answers | noul | **understudent-deep-split** | 23.0% | 93.2% | 94.0% | 98723 | understudent-ladder-split (23.0%) |
| minimal_pair | payment_flip | noul | **understudent-lite-split** | 58.5% | 100.0% | 100.0% | 150 | understudent-keywords-split (41.5%) |
| pair_compare | politer_reply | choice | **understudent-probe-split** | 96.0% | 99.2% | 99.8% | 73092 | understudent-deep-split (96.0%) |
| conditional_rule | retry_exception | noul | **understudent-keywords-split** | 80.0% | 99.5% | 100.0% | 420 | understudent-lite-split (35.5%) |
| needle | secret_needle | noul | none | | | | | |
| surface_property | tool_failed | noul | **understudent-deep-split** | 84.5% | 92.5% | 92.0% | 145080 | understudent-probe-split (78.0%) |

## Coverage by readout (from each system's summary)

| dataset | system | coverage_by_readout |
|---|---|---|
| output_kind | understudent-probe | mean 57.0% |
| output_kind | understudent-deep | pool 63.7% |
| output_kind | understudent-keywords | keywords 51.0% |
| output_kind | understudent-lite | lite 52.2% |
| output_kind | understudent-neighbors | neighbors 43.8% |
| output_kind | understudent-ladder | keywords 41.5%, deep:pool 23.3% |
| output_kind | understudent-probe-split | mean 83.5% |
| output_kind | understudent-deep-split | pool 91.0% |
| output_kind | understudent-keywords-split | keywords 75.5% |
| output_kind | understudent-lite-split | lite 81.5% |
| output_kind | understudent-neighbors-split | neighbors 65.8% |
| output_kind | understudent-ladder-split | keywords 72.2%, deep:pool 21.8% |
| passage_answers | understudent-deep | pool 15.7% |
| passage_answers | understudent-ladder | deep:pool 15.7% |
| passage_answers | understudent-deep-split | pool 23.5% |
| passage_answers | understudent-ladder-split | deep:pool 23.5% |
| payment_flip | understudent-probe | mean 47.2% |
| payment_flip | understudent-deep | pool 69.7% |
| payment_flip | understudent-keywords | keywords 13.8% |
| payment_flip | understudent-lite | lite 24.8% |
| payment_flip | understudent-ladder | deep:pool 69.7% |
| payment_flip | understudent-probe-split | mean 73.0% |
| payment_flip | understudent-deep-split | pool 88.8% |
| payment_flip | understudent-keywords-split | keywords 20.8% |
| payment_flip | understudent-lite-split | lite 37.2% |
| payment_flip | understudent-ladder-split | deep:pool 88.8% |
| politer_reply | understudent-probe | mean 78.2% |
| politer_reply | understudent-deep | pool 78.2% |
| politer_reply | understudent-keywords | keywords 80.0% |
| politer_reply | understudent-lite | lite 73.0% |
| politer_reply | understudent-neighbors | neighbors 63.8% |
| politer_reply | understudent-ladder | keywords 80.0%, deep:pool 2.7% |
| politer_reply | understudent-probe-split | mean 95.8% |
| politer_reply | understudent-deep-split | attn 95.8% |
| politer_reply | understudent-keywords-split | keywords 97.5% |
| politer_reply | understudent-lite-split | lite 93.0% |
| politer_reply | understudent-neighbors-split | neighbors 95.8% |
| politer_reply | understudent-ladder-split | keywords 97.5%, deep:attn 2.2% |
| retry_exception | understudent-probe | mean 50.8% |
| retry_exception | understudent-deep | pool 49.7% |
| retry_exception | understudent-keywords | keywords 26.7% |
| retry_exception | understudent-lite | lite 11.8% |
| retry_exception | understudent-ladder | keywords 15.0%, deep:pool 36.2% |
| retry_exception | understudent-probe-split | mean 76.2% |
| retry_exception | understudent-deep-split | pool 74.5% |
| retry_exception | understudent-keywords-split | keywords 40.0% |
| retry_exception | understudent-lite-split | lite 17.8% |
| retry_exception | understudent-ladder-split | keywords 22.5%, deep:pool 54.2% |
| tool_failed | understudent-probe | mean 39.2% |
| tool_failed | understudent-deep | pool 28.2% |
| tool_failed | understudent-keywords | keywords 32.3% |
| tool_failed | understudent-ladder | keywords 32.3%, deep:pool 2.3% |
| tool_failed | understudent-probe-split | mean 58.8% |
| tool_failed | understudent-deep-split | pool 42.2% |
| tool_failed | understudent-keywords-split | keywords 48.5% |
| tool_failed | understudent-ladder-split | keywords 48.5%, deep:pool 3.5% |

## Summary by wire type

| type | system | datasets | takes over | partial | correctly abstains | harmful | mean cov. 2nd half | mean Δacc vs lead |
|---|---|---|---|---|---|---|---|---|
| choice | understudent-ladder | 3 | 1 | 0 | 1 | 1 | 59.9% | -1.6 pt |
| choice | understudent-ladder-split | 3 | 1 | 0 | 1 | 1 | 64.5% | -1.4 pt |
| choice | understudent-neighbors | 3 | 1 | 0 | 1 | 1 | 54.8% | -1.2 pt |
| choice | understudent-neighbors-split | 3 | 0 | 1 | 1 | 1 | 56.0% | -1.8 pt |
| choice | understudent-keywords | 3 | 1 | 0 | 1 | 1 | 55.8% | -1.5 pt |
| choice | understudent-keywords-split | 3 | 1 | 0 | 1 | 1 | 57.5% | -1.2 pt |
| choice | understudent-probe | 3 | 2 | 0 | 1 | 0 | 58.1% | -0.3 pt |
| choice | understudent-probe-split | 3 | 1 | 1 | 1 | 0 | 60.0% | -0.7 pt |
| choice | understudent-deep | 3 | 1 | 1 | 1 | 0 | 58.7% | -0.6 pt |
| choice | understudent-deep-split | 3 | 1 | 1 | 1 | 0 | 62.2% | -0.6 pt |
| choice | understudent-lite | 3 | 1 | 1 | 1 | 0 | 51.9% | -0.9 pt |
| choice | understudent-lite-split | 3 | 1 | 0 | 1 | 1 | 58.7% | -1.2 pt |
| choice | hunch | 3 | 1 | 0 | 2 | 0 | 26.3% | -0.3 pt |
| choice | hunch-split | 3 | 0 | 1 | 2 | 0 | 28.0% | -0.5 pt |
| noul | understudent-ladder | 6 | 1 | 2 | 2 | 1 | 44.5% | -1.0 pt |
| noul | understudent-ladder-split | 6 | 1 | 2 | 2 | 1 | 45.5% | -0.7 pt |
| noul | understudent-neighbors | 6 | 0 | 0 | 6 | 0 | 0.0% | +0.0 pt |
| noul | understudent-neighbors-split | 6 | 0 | 0 | 6 | 0 | 0.0% | +0.0 pt |
| noul | understudent-keywords | 6 | 2 | 1 | 3 | 0 | 24.3% | +0.0 pt |
| noul | understudent-keywords-split | 6 | 2 | 1 | 3 | 0 | 31.2% | +0.0 pt |
| noul | understudent-probe | 6 | 1 | 0 | 3 | 2 | 34.0% | -1.2 pt |
| noul | understudent-probe-split | 6 | 1 | 0 | 3 | 2 | 39.6% | -1.0 pt |
| noul | understudent-deep | 6 | 1 | 2 | 2 | 1 | 41.8% | -1.1 pt |
| noul | understudent-deep-split | 6 | 1 | 2 | 2 | 1 | 46.7% | -0.8 pt |
| noul | understudent-lite | 6 | 0 | 2 | 4 | 0 | 12.2% | -0.1 pt |
| noul | understudent-lite-split | 6 | 1 | 1 | 4 | 0 | 15.7% | -0.1 pt |
| noul | hunch | 6 | 1 | 0 | 3 | 2 | 34.2% | -1.3 pt |
| noul | hunch-split | 6 | 1 | 0 | 3 | 2 | 39.5% | -1.1 pt |
| score | understudent-ladder | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-ladder-split | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-neighbors | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-neighbors-split | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-keywords | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-keywords-split | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-probe | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-probe-split | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-deep | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-deep-split | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-lite | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-lite-split | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | hunch | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | hunch-split | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |

## Warm start vs cold start (operate items only: stream items 201..end)

Warm start = seeded with 200 lead answers via train, then judging. Cold start = the replay's decisions on the same items.

| dataset | category | system | graduated right after training | coverage | cov. 2nd half | acc lead | acc system | kappa | verdict |
|---|---|---|---|---|---|---|---|---|---|
| capital_city | world_knowledge | hunch (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.8% | – | correctly abstains |
| capital_city | world_knowledge | understudent-probe (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| capital_city | world_knowledge | understudent-deep (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| capital_city | world_knowledge | understudent-keywords (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| capital_city | world_knowledge | understudent-lite (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| capital_city | world_knowledge | understudent-neighbors (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| capital_city | world_knowledge | understudent-ladder (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| capital_city | world_knowledge | hunch-split | no | 0.0% | 0.0% | 99.5% | 99.8% | – | correctly abstains |
| capital_city | world_knowledge | understudent-probe-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| capital_city | world_knowledge | understudent-deep-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| capital_city | world_knowledge | understudent-keywords-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| capital_city | world_knowledge | understudent-lite-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| capital_city | world_knowledge | understudent-neighbors-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| capital_city | world_knowledge | understudent-ladder-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| code_lookup | binding_lookup | hunch (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| code_lookup | binding_lookup | understudent-probe (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| code_lookup | binding_lookup | understudent-deep (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| code_lookup | binding_lookup | understudent-keywords (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| code_lookup | binding_lookup | understudent-lite (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| code_lookup | binding_lookup | understudent-neighbors (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| code_lookup | binding_lookup | understudent-ladder (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| code_lookup | binding_lookup | hunch-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| code_lookup | binding_lookup | understudent-probe-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| code_lookup | binding_lookup | understudent-deep-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| code_lookup | binding_lookup | understudent-keywords-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| code_lookup | binding_lookup | understudent-lite-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| code_lookup | binding_lookup | understudent-neighbors-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| code_lookup | binding_lookup | understudent-ladder-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| customer_frustration | graded_tone | hunch (cold, same items) | – | 0.0% | 0.0% | 67.2% | 67.2% | – | correctly abstains |
| customer_frustration | graded_tone | understudent-probe (cold, same items) | – | 0.0% | 0.0% | 67.2% | 67.2% | – | correctly abstains |
| customer_frustration | graded_tone | understudent-deep (cold, same items) | – | 0.0% | 0.0% | 67.2% | 67.2% | – | correctly abstains |
| customer_frustration | graded_tone | understudent-keywords (cold, same items) | – | 0.0% | 0.0% | 67.2% | 67.2% | – | correctly abstains |
| customer_frustration | graded_tone | understudent-lite (cold, same items) | – | 0.0% | 0.0% | 67.2% | 67.2% | – | correctly abstains |
| customer_frustration | graded_tone | understudent-neighbors (cold, same items) | – | 0.0% | 0.0% | 67.2% | 67.2% | – | correctly abstains |
| customer_frustration | graded_tone | understudent-ladder (cold, same items) | – | 0.0% | 0.0% | 67.2% | 67.2% | – | correctly abstains |
| customer_frustration | graded_tone | hunch-split | no | 0.0% | 0.0% | 67.2% | 67.2% | – | correctly abstains |
| customer_frustration | graded_tone | understudent-probe-split | no | 0.0% | 0.0% | 67.2% | 67.2% | – | correctly abstains |
| customer_frustration | graded_tone | understudent-deep-split | no | 0.0% | 0.0% | 67.2% | 67.2% | – | correctly abstains |
| customer_frustration | graded_tone | understudent-keywords-split | no | 0.0% | 0.0% | 67.2% | 67.2% | – | correctly abstains |
| customer_frustration | graded_tone | understudent-lite-split | no | 0.0% | 0.0% | 67.2% | 67.2% | – | correctly abstains |
| customer_frustration | graded_tone | understudent-neighbors-split | no | 0.0% | 0.0% | 67.2% | 67.2% | – | correctly abstains |
| customer_frustration | graded_tone | understudent-ladder-split | no | 0.0% | 0.0% | 67.2% | 67.2% | – | correctly abstains |
| incident_severity | compound_judgment | hunch (cold, same items) | – | 0.0% | 0.0% | 67.0% | 67.0% | – | correctly abstains |
| incident_severity | compound_judgment | understudent-probe (cold, same items) | – | 0.0% | 0.0% | 67.0% | 67.0% | – | correctly abstains |
| incident_severity | compound_judgment | understudent-deep (cold, same items) | – | 0.0% | 0.0% | 67.0% | 67.0% | – | correctly abstains |
| incident_severity | compound_judgment | understudent-keywords (cold, same items) | – | 0.0% | 0.0% | 67.0% | 67.0% | – | correctly abstains |
| incident_severity | compound_judgment | understudent-lite (cold, same items) | – | 0.0% | 0.0% | 67.0% | 67.0% | – | correctly abstains |
| incident_severity | compound_judgment | understudent-neighbors (cold, same items) | – | 0.0% | 0.0% | 67.0% | 67.0% | – | correctly abstains |
| incident_severity | compound_judgment | understudent-ladder (cold, same items) | – | 0.0% | 0.0% | 67.0% | 67.0% | – | correctly abstains |
| incident_severity | compound_judgment | hunch-split | no | 0.0% | 0.0% | 67.0% | 67.0% | – | correctly abstains |
| incident_severity | compound_judgment | understudent-probe-split | no | 0.0% | 0.0% | 67.0% | 67.0% | – | correctly abstains |
| incident_severity | compound_judgment | understudent-deep-split | no | 0.0% | 0.0% | 67.0% | 67.0% | – | correctly abstains |
| incident_severity | compound_judgment | understudent-keywords-split | no | 0.0% | 0.0% | 67.0% | 67.0% | – | correctly abstains |
| incident_severity | compound_judgment | understudent-lite-split | no | 0.0% | 0.0% | 67.0% | 67.0% | – | correctly abstains |
| incident_severity | compound_judgment | understudent-neighbors-split | no | 0.0% | 0.0% | 67.0% | 67.0% | – | correctly abstains |
| incident_severity | compound_judgment | understudent-ladder-split | no | 0.0% | 0.0% | 67.0% | 67.0% | – | correctly abstains |
| output_kind | topic_kind | hunch (cold, same items) | – | 74.0% | 82.5% | 96.5% | 95.2% | 0.97 | partial |
| output_kind | topic_kind | understudent-probe (cold, same items) | – | 74.0% | 82.5% | 96.5% | 95.2% | 0.97 | partial |
| output_kind | topic_kind | understudent-deep (cold, same items) | – | 78.5% | 83.5% | 96.5% | 96.2% | 0.97 | takes over |
| output_kind | topic_kind | understudent-keywords (cold, same items) | – | 65.5% | 70.0% | 96.5% | 96.8% | 0.97 | takes over |
| output_kind | topic_kind | understudent-lite (cold, same items) | – | 64.5% | 66.5% | 96.5% | 95.8% | 0.95 | takes over |
| output_kind | topic_kind | understudent-neighbors (cold, same items) | – | 65.8% | 72.0% | 96.5% | 95.0% | 0.97 | partial |
| output_kind | topic_kind | understudent-ladder (cold, same items) | – | 79.2% | 80.0% | 96.5% | 96.5% | 0.96 | takes over |
| output_kind | topic_kind | hunch-split | yes | 83.5% | 84.0% | 96.5% | 95.0% | 0.95 | partial |
| output_kind | topic_kind | understudent-probe-split | yes | 83.5% | 84.0% | 96.5% | 95.0% | 0.95 | partial |
| output_kind | topic_kind | understudent-deep-split | yes | 91.0% | 90.5% | 96.5% | 95.2% | 0.94 | partial |
| output_kind | topic_kind | understudent-keywords-split | yes | 75.5% | 75.0% | 96.5% | 96.8% | 0.97 | takes over |
| output_kind | topic_kind | understudent-lite-split | yes | 81.5% | 81.0% | 96.5% | 94.0% | 0.92 | harmful |
| output_kind | topic_kind | understudent-neighbors-split | yes | 65.8% | 72.0% | 96.5% | 95.0% | 0.97 | partial |
| output_kind | topic_kind | understudent-ladder-split | yes | 94.0% | 93.5% | 96.5% | 96.0% | 0.94 | takes over |
| passage_answers | relational_match | hunch (cold, same items) | – | 0.0% | 0.0% | 94.0% | 94.2% | – | correctly abstains |
| passage_answers | relational_match | understudent-probe (cold, same items) | – | 0.0% | 0.0% | 94.0% | 94.0% | – | correctly abstains |
| passage_answers | relational_match | understudent-deep (cold, same items) | – | 23.5% | 23.0% | 94.0% | 93.2% | 0.84 | partial |
| passage_answers | relational_match | understudent-keywords (cold, same items) | – | 0.0% | 0.0% | 94.0% | 94.0% | – | correctly abstains |
| passage_answers | relational_match | understudent-lite (cold, same items) | – | 0.0% | 0.0% | 94.0% | 94.0% | – | correctly abstains |
| passage_answers | relational_match | understudent-neighbors (cold, same items) | – | 0.0% | 0.0% | 94.0% | 94.0% | – | correctly abstains |
| passage_answers | relational_match | understudent-ladder (cold, same items) | – | 23.5% | 23.0% | 94.0% | 93.2% | 0.84 | partial |
| passage_answers | relational_match | hunch-split | no | 0.0% | 0.0% | 94.0% | 94.2% | – | correctly abstains |
| passage_answers | relational_match | understudent-probe-split | no | 0.0% | 0.0% | 94.0% | 94.0% | – | correctly abstains |
| passage_answers | relational_match | understudent-deep-split | no | 23.5% | 23.0% | 94.0% | 93.2% | 0.84 | partial |
| passage_answers | relational_match | understudent-keywords-split | no | 0.0% | 0.0% | 94.0% | 94.0% | – | correctly abstains |
| passage_answers | relational_match | understudent-lite-split | no | 0.0% | 0.0% | 94.0% | 94.0% | – | correctly abstains |
| passage_answers | relational_match | understudent-neighbors-split | no | 0.0% | 0.0% | 94.0% | 94.0% | – | correctly abstains |
| passage_answers | relational_match | understudent-ladder-split | no | 23.5% | 23.0% | 94.0% | 93.2% | 0.84 | partial |
| payment_flip | minimal_pair | hunch (cold, same items) | – | 53.0% | 55.0% | 100.0% | 95.2% | 0.79 | harmful |
| payment_flip | minimal_pair | understudent-probe (cold, same items) | – | 53.0% | 55.0% | 100.0% | 95.2% | 0.79 | harmful |
| payment_flip | minimal_pair | understudent-deep (cold, same items) | – | 84.8% | 92.0% | 100.0% | 94.0% | 0.83 | harmful |
| payment_flip | minimal_pair | understudent-keywords (cold, same items) | – | 20.8% | 41.5% | 100.0% | 99.8% | 0.97 | partial |
| payment_flip | minimal_pair | understudent-lite (cold, same items) | – | 37.2% | 58.5% | 100.0% | 100.0% | 1.00 | takes over |
| payment_flip | minimal_pair | understudent-neighbors (cold, same items) | – | 0.0% | 0.0% | 100.0% | 100.0% | – | correctly abstains |
| payment_flip | minimal_pair | understudent-ladder (cold, same items) | – | 84.8% | 92.0% | 100.0% | 94.0% | 0.83 | harmful |
| payment_flip | minimal_pair | hunch-split | yes | 73.0% | 80.0% | 100.0% | 97.5% | 0.91 | harmful |
| payment_flip | minimal_pair | understudent-probe-split | yes | 73.0% | 80.0% | 100.0% | 97.5% | 0.91 | harmful |
| payment_flip | minimal_pair | understudent-deep-split | yes | 88.8% | 90.0% | 100.0% | 97.2% | 0.92 | harmful |
| payment_flip | minimal_pair | understudent-keywords-split | no | 20.8% | 41.5% | 100.0% | 99.8% | 0.97 | partial |
| payment_flip | minimal_pair | understudent-lite-split | no | 37.2% | 58.5% | 100.0% | 100.0% | 1.00 | takes over |
| payment_flip | minimal_pair | understudent-neighbors-split | no | 0.0% | 0.0% | 100.0% | 100.0% | – | correctly abstains |
| payment_flip | minimal_pair | understudent-ladder-split | yes | 88.8% | 90.0% | 100.0% | 97.2% | 0.92 | harmful |
| politer_reply | pair_compare | hunch (cold, same items) | – | 0.0% | 0.0% | 99.8% | 99.8% | – | correctly abstains |
| politer_reply | pair_compare | understudent-probe (cold, same items) | – | 94.8% | 95.0% | 99.8% | 99.8% | 1.00 | takes over |
| politer_reply | pair_compare | understudent-deep (cold, same items) | – | 94.8% | 95.0% | 99.8% | 98.5% | 0.97 | partial |
| politer_reply | pair_compare | understudent-keywords (cold, same items) | – | 96.8% | 97.5% | 99.8% | 94.2% | 0.89 | harmful |
| politer_reply | pair_compare | understudent-lite (cold, same items) | – | 88.5% | 91.5% | 99.8% | 97.8% | 0.95 | partial |
| politer_reply | pair_compare | understudent-neighbors (cold, same items) | – | 95.8% | 96.0% | 99.8% | 95.8% | 0.92 | harmful |
| politer_reply | pair_compare | understudent-ladder (cold, same items) | – | 99.0% | 99.0% | 99.8% | 94.2% | 0.89 | harmful |
| politer_reply | pair_compare | hunch-split | no | 0.0% | 0.0% | 99.8% | 99.8% | – | correctly abstains |
| politer_reply | pair_compare | understudent-probe-split | yes | 95.8% | 96.0% | 99.8% | 99.2% | 0.99 | takes over |
| politer_reply | pair_compare | understudent-deep-split | yes | 95.8% | 96.0% | 99.8% | 99.2% | 0.99 | takes over |
| politer_reply | pair_compare | understudent-keywords-split | yes | 97.5% | 97.5% | 99.8% | 96.0% | 0.92 | harmful |
| politer_reply | pair_compare | understudent-lite-split | yes | 93.0% | 95.0% | 99.8% | 98.8% | 0.98 | takes over |
| politer_reply | pair_compare | understudent-neighbors-split | yes | 95.8% | 96.0% | 99.8% | 95.8% | 0.92 | harmful |
| politer_reply | pair_compare | understudent-ladder-split | yes | 99.8% | 100.0% | 99.8% | 96.0% | 0.92 | harmful |
| retry_exception | conditional_rule | hunch (cold, same items) | – | 79.2% | 79.0% | 100.0% | 94.8% | 0.81 | harmful |
| retry_exception | conditional_rule | understudent-probe (cold, same items) | – | 76.2% | 79.5% | 100.0% | 95.8% | 0.83 | harmful |
| retry_exception | conditional_rule | understudent-deep (cold, same items) | – | 74.5% | 82.5% | 100.0% | 98.2% | 0.93 | partial |
| retry_exception | conditional_rule | understudent-keywords (cold, same items) | – | 40.0% | 80.0% | 100.0% | 99.5% | 0.96 | takes over |
| retry_exception | conditional_rule | understudent-lite (cold, same items) | – | 17.8% | 35.5% | 100.0% | 99.2% | 0.84 | partial |
| retry_exception | conditional_rule | understudent-neighbors (cold, same items) | – | 0.0% | 0.0% | 100.0% | 100.0% | – | correctly abstains |
| retry_exception | conditional_rule | understudent-ladder (cold, same items) | – | 76.8% | 87.0% | 100.0% | 98.2% | 0.93 | partial |
| retry_exception | conditional_rule | hunch-split | yes | 79.2% | 79.0% | 100.0% | 94.8% | 0.81 | harmful |
| retry_exception | conditional_rule | understudent-probe-split | yes | 76.2% | 79.5% | 100.0% | 95.8% | 0.83 | harmful |
| retry_exception | conditional_rule | understudent-deep-split | yes | 74.5% | 82.5% | 100.0% | 98.2% | 0.93 | partial |
| retry_exception | conditional_rule | understudent-keywords-split | no | 40.0% | 80.0% | 100.0% | 99.5% | 0.96 | takes over |
| retry_exception | conditional_rule | understudent-lite-split | no | 17.8% | 35.5% | 100.0% | 99.2% | 0.84 | partial |
| retry_exception | conditional_rule | understudent-neighbors-split | no | 0.0% | 0.0% | 100.0% | 100.0% | – | correctly abstains |
| retry_exception | conditional_rule | understudent-ladder-split | yes | 76.8% | 87.0% | 100.0% | 98.2% | 0.93 | partial |
| secret_needle | needle | hunch (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| secret_needle | needle | understudent-probe (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| secret_needle | needle | understudent-deep (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| secret_needle | needle | understudent-keywords (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| secret_needle | needle | understudent-lite (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| secret_needle | needle | understudent-neighbors (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| secret_needle | needle | understudent-ladder (cold, same items) | – | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| secret_needle | needle | hunch-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| secret_needle | needle | understudent-probe-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| secret_needle | needle | understudent-deep-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| secret_needle | needle | understudent-keywords-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| secret_needle | needle | understudent-lite-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| secret_needle | needle | understudent-neighbors-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| secret_needle | needle | understudent-ladder-split | no | 0.0% | 0.0% | 99.5% | 99.5% | – | correctly abstains |
| tool_failed | surface_property | hunch (cold, same items) | – | 58.8% | 78.0% | 92.0% | 92.5% | 0.92 | takes over |
| tool_failed | surface_property | understudent-probe (cold, same items) | – | 58.8% | 78.0% | 92.0% | 92.5% | 0.92 | takes over |
| tool_failed | surface_property | understudent-deep (cold, same items) | – | 42.2% | 84.5% | 92.0% | 92.5% | 0.93 | takes over |
| tool_failed | surface_property | understudent-keywords (cold, same items) | – | 48.5% | 66.0% | 92.0% | 92.8% | 0.82 | takes over |
| tool_failed | surface_property | understudent-lite (cold, same items) | – | 0.0% | 0.0% | 92.0% | 92.0% | – | correctly abstains |
| tool_failed | surface_property | understudent-neighbors (cold, same items) | – | 0.0% | 0.0% | 92.0% | 92.0% | – | correctly abstains |
| tool_failed | surface_property | understudent-ladder (cold, same items) | – | 52.0% | 73.0% | 92.0% | 92.8% | 0.78 | takes over |
| tool_failed | surface_property | hunch-split | no | 58.8% | 78.0% | 92.0% | 92.5% | 0.92 | takes over |
| tool_failed | surface_property | understudent-probe-split | no | 58.8% | 78.0% | 92.0% | 92.5% | 0.92 | takes over |
| tool_failed | surface_property | understudent-deep-split | no | 42.2% | 84.5% | 92.0% | 92.5% | 0.93 | takes over |
| tool_failed | surface_property | understudent-keywords-split | no | 48.5% | 66.0% | 92.0% | 92.8% | 0.82 | takes over |
| tool_failed | surface_property | understudent-lite-split | no | 0.0% | 0.0% | 92.0% | 92.0% | – | correctly abstains |
| tool_failed | surface_property | understudent-neighbors-split | no | 0.0% | 0.0% | 92.0% | 92.0% | – | correctly abstains |
| tool_failed | surface_property | understudent-ladder-split | no | 52.0% | 73.0% | 92.0% | 92.8% | 0.78 | takes over |

## Timing

| dataset | system | wall s | train s | items | status |
|---|---|---|---|---|---|
| capital_city | hunch | 25.0 | – | 600 | ok |
| capital_city | understudent-probe | 14.6 | – | 600 | ok |
| capital_city | understudent-deep | 23.0 | – | 600 | ok |
| capital_city | understudent-keywords | 0.2 | – | 600 | ok |
| capital_city | understudent-lite | 1.9 | – | 600 | ok |
| capital_city | understudent-neighbors | 12.4 | – | 600 | ok |
| capital_city | understudent-ladder | 21.0 | – | 600 | ok |
| capital_city | hunch-split | 19.2 | – | 600 | ok |
| capital_city | understudent-probe-split | 11.7 | 3.0 | 600 | ok |
| capital_city | understudent-deep-split | 20.5 | 4.4 | 600 | ok |
| capital_city | understudent-keywords-split | 0.2 | 0.0 | 600 | ok |
| capital_city | understudent-lite-split | 3.5 | 1.7 | 600 | ok |
| capital_city | understudent-neighbors-split | 11.8 | 2.9 | 600 | ok |
| capital_city | understudent-ladder-split | 19.8 | 4.3 | 600 | ok |
| code_lookup | hunch | 42.1 | – | 600 | ok |
| code_lookup | understudent-probe | 28.9 | – | 600 | ok |
| code_lookup | understudent-deep | 53.2 | – | 600 | ok |
| code_lookup | understudent-keywords | 0.7 | – | 600 | ok |
| code_lookup | understudent-lite | 2.6 | – | 600 | ok |
| code_lookup | understudent-neighbors | 33.3 | – | 600 | ok |
| code_lookup | understudent-ladder | 61.6 | – | 600 | ok |
| code_lookup | hunch-split | 45.9 | – | 600 | ok |
| code_lookup | understudent-probe-split | 26.6 | 6.1 | 600 | ok |
| code_lookup | understudent-deep-split | 51.1 | 11.3 | 600 | ok |
| code_lookup | understudent-keywords-split | 0.8 | 0.1 | 600 | ok |
| code_lookup | understudent-lite-split | 3.9 | 1.8 | 600 | ok |
| code_lookup | understudent-neighbors-split | 25.9 | 6.9 | 600 | ok |
| code_lookup | understudent-ladder-split | 45.4 | 8.6 | 600 | ok |
| customer_frustration | hunch | 65.2 | – | 600 | ok |
| customer_frustration | understudent-probe | 42.0 | – | 600 | ok |
| customer_frustration | understudent-deep | 70.7 | – | 600 | ok |
| customer_frustration | understudent-keywords | 3.5 | – | 600 | ok |
| customer_frustration | understudent-lite | 2.0 | – | 600 | ok |
| customer_frustration | understudent-neighbors | 39.7 | – | 600 | ok |
| customer_frustration | understudent-ladder | 74.1 | – | 600 | ok |
| customer_frustration | hunch-split | 55.1 | – | 600 | ok |
| customer_frustration | understudent-probe-split | 36.7 | 9.8 | 600 | ok |
| customer_frustration | understudent-deep-split | 62.9 | 12.8 | 600 | ok |
| customer_frustration | understudent-keywords-split | 3.1 | 0.4 | 600 | ok |
| customer_frustration | understudent-lite-split | 3.5 | 1.5 | 600 | ok |
| customer_frustration | understudent-neighbors-split | 39.2 | 8.4 | 600 | ok |
| customer_frustration | understudent-ladder-split | 80.4 | 15.5 | 600 | ok |
| incident_severity | hunch | 109.0 | – | 600 | ok |
| incident_severity | understudent-probe | 71.2 | – | 600 | ok |
| incident_severity | understudent-deep | 136.5 | – | 600 | ok |
| incident_severity | understudent-keywords | 4.8 | – | 600 | ok |
| incident_severity | understudent-lite | 2.3 | – | 600 | ok |
| incident_severity | understudent-neighbors | 81.0 | – | 600 | ok |
| incident_severity | understudent-ladder | 121.4 | – | 600 | ok |
| incident_severity | hunch-split | 66.7 | – | 600 | ok |
| incident_severity | understudent-probe-split | 57.9 | 13.2 | 600 | ok |
| incident_severity | understudent-deep-split | 116.6 | 25.9 | 600 | ok |
| incident_severity | understudent-keywords-split | 4.6 | 0.6 | 600 | ok |
| incident_severity | understudent-lite-split | 3.7 | 1.7 | 600 | ok |
| incident_severity | understudent-neighbors-split | 60.5 | 15.2 | 600 | ok |
| incident_severity | understudent-ladder-split | 120.8 | 24.2 | 600 | ok |
| output_kind | hunch | 71.2 | – | 600 | ok |
| output_kind | understudent-probe | 56.2 | – | 600 | ok |
| output_kind | understudent-deep | 92.2 | – | 600 | ok |
| output_kind | understudent-keywords | 1.2 | – | 600 | ok |
| output_kind | understudent-lite | 1.7 | – | 600 | ok |
| output_kind | understudent-neighbors | 61.6 | – | 600 | ok |
| output_kind | understudent-ladder | 69.1 | – | 600 | ok |
| output_kind | hunch-split | 57.6 | – | 600 | ok |
| output_kind | understudent-probe-split | 53.4 | 13.2 | 600 | ok |
| output_kind | understudent-deep-split | 79.1 | 20.7 | 600 | ok |
| output_kind | understudent-keywords-split | 1.1 | 0.4 | 600 | ok |
| output_kind | understudent-lite-split | 3.4 | 1.6 | 600 | ok |
| output_kind | understudent-neighbors-split | 60.6 | 13.6 | 600 | ok |
| output_kind | understudent-ladder-split | 47.8 | 24.4 | 600 | ok |
| passage_answers | hunch | 73.9 | – | 600 | ok |
| passage_answers | understudent-probe | 49.0 | – | 600 | ok |
| passage_answers | understudent-deep | 78.0 | – | 600 | ok |
| passage_answers | understudent-keywords | 1.2 | – | 600 | ok |
| passage_answers | understudent-lite | 2.2 | – | 600 | ok |
| passage_answers | understudent-neighbors | 53.2 | – | 600 | ok |
| passage_answers | understudent-ladder | 86.3 | – | 600 | ok |
| passage_answers | hunch-split | 53.1 | – | 600 | ok |
| passage_answers | understudent-probe-split | 41.5 | 12.2 | 600 | ok |
| passage_answers | understudent-deep-split | 73.1 | 18.4 | 600 | ok |
| passage_answers | understudent-keywords-split | 1.6 | 0.3 | 600 | ok |
| passage_answers | understudent-lite-split | 4.1 | 1.6 | 600 | ok |
| passage_answers | understudent-neighbors-split | 46.5 | 11.3 | 600 | ok |
| passage_answers | understudent-ladder-split | 81.7 | 20.4 | 600 | ok |
| payment_flip | hunch | 25.3 | – | 600 | ok |
| payment_flip | understudent-probe | 26.3 | – | 600 | ok |
| payment_flip | understudent-deep | 38.1 | – | 600 | ok |
| payment_flip | understudent-keywords | 0.2 | – | 600 | ok |
| payment_flip | understudent-lite | 1.8 | – | 600 | ok |
| payment_flip | understudent-neighbors | 24.6 | – | 600 | ok |
| payment_flip | understudent-ladder | 38.4 | – | 600 | ok |
| payment_flip | hunch-split | 27.8 | – | 600 | ok |
| payment_flip | understudent-probe-split | 24.3 | 6.2 | 600 | ok |
| payment_flip | understudent-deep-split | 34.8 | 8.8 | 600 | ok |
| payment_flip | understudent-keywords-split | 0.3 | 0.1 | 600 | ok |
| payment_flip | understudent-lite-split | 3.1 | 1.4 | 600 | ok |
| payment_flip | understudent-neighbors-split | 21.2 | 5.2 | 600 | ok |
| payment_flip | understudent-ladder-split | 34.2 | 7.6 | 600 | ok |
| politer_reply | hunch | 67.4 | – | 600 | ok |
| politer_reply | understudent-probe | 48.0 | – | 600 | ok |
| politer_reply | understudent-deep | 70.9 | – | 600 | ok |
| politer_reply | understudent-keywords | 0.2 | – | 600 | ok |
| politer_reply | understudent-lite | 1.9 | – | 600 | ok |
| politer_reply | understudent-neighbors | 50.5 | – | 600 | ok |
| politer_reply | understudent-ladder | 20.5 | – | 600 | ok |
| politer_reply | hunch-split | 52.4 | – | 600 | ok |
| politer_reply | understudent-probe-split | 41.9 | 11.2 | 600 | ok |
| politer_reply | understudent-deep-split | 60.3 | 17.5 | 600 | ok |
| politer_reply | understudent-keywords-split | 0.3 | 0.1 | 600 | ok |
| politer_reply | understudent-lite-split | 2.6 | 1.3 | 600 | ok |
| politer_reply | understudent-neighbors-split | 40.2 | 10.7 | 600 | ok |
| politer_reply | understudent-ladder-split | 28.0 | 20.9 | 600 | ok |
| retry_exception | hunch | 201.6 | – | 600 | ok |
| retry_exception | understudent-probe | 193.1 | – | 600 | ok |
| retry_exception | understudent-deep | 340.8 | – | 600 | ok |
| retry_exception | understudent-keywords | 0.9 | – | 600 | ok |
| retry_exception | understudent-lite | 2.0 | – | 600 | ok |
| retry_exception | understudent-neighbors | 250.4 | – | 600 | ok |
| retry_exception | understudent-ladder | 291.9 | – | 600 | ok |
| retry_exception | hunch-split | 195.4 | – | 600 | ok |
| retry_exception | understudent-probe-split | 192.7 | 48.8 | 600 | ok |
| retry_exception | understudent-deep-split | 296.1 | 79.2 | 600 | ok |
| retry_exception | understudent-keywords-split | 0.9 | 0.2 | 600 | ok |
| retry_exception | understudent-lite-split | 3.4 | 1.6 | 600 | ok |
| retry_exception | understudent-neighbors-split | 190.1 | 52.4 | 600 | ok |
| retry_exception | understudent-ladder-split | 257.6 | 84.8 | 600 | ok |
| secret_needle | hunch | 377.5 | – | 600 | ok |
| secret_needle | understudent-probe | 366.5 | – | 600 | ok |
| secret_needle | understudent-deep | 586.2 | – | 600 | ok |
| secret_needle | understudent-keywords | 3.1 | – | 600 | ok |
| secret_needle | understudent-lite | 2.7 | – | 600 | ok |
| secret_needle | understudent-neighbors | 360.1 | – | 600 | ok |
| secret_needle | understudent-ladder | 704.2 | – | 600 | ok |
| secret_needle | hunch-split | 365.4 | – | 600 | ok |
| secret_needle | understudent-probe-split | 332.3 | 96.7 | 600 | ok |
| secret_needle | understudent-deep-split | 498.6 | 138.5 | 600 | ok |
| secret_needle | understudent-keywords-split | 2.7 | 0.4 | 600 | ok |
| secret_needle | understudent-lite-split | 3.9 | 1.6 | 600 | ok |
| secret_needle | understudent-neighbors-split | 301.1 | 77.6 | 600 | ok |
| secret_needle | understudent-ladder-split | 581.0 | 151.1 | 600 | ok |
| tool_failed | hunch | 81.5 | – | 600 | ok |
| tool_failed | understudent-probe | 76.5 | – | 600 | ok |
| tool_failed | understudent-deep | 116.4 | – | 600 | ok |
| tool_failed | understudent-keywords | 0.7 | – | 600 | ok |
| tool_failed | understudent-lite | 2.2 | – | 600 | ok |
| tool_failed | understudent-neighbors | 67.9 | – | 600 | ok |
| tool_failed | understudent-ladder | 82.1 | – | 600 | ok |
| tool_failed | hunch-split | 65.3 | – | 600 | ok |
| tool_failed | understudent-probe-split | 58.4 | 13.7 | 600 | ok |
| tool_failed | understudent-deep-split | 92.6 | 23.4 | 600 | ok |
| tool_failed | understudent-keywords-split | 0.7 | 0.1 | 600 | ok |
| tool_failed | understudent-lite-split | 4.4 | 2.0 | 600 | ok |
| tool_failed | understudent-neighbors-split | 62.4 | 14.5 | 600 | ok |
| tool_failed | understudent-ladder-split | 71.3 | 26.1 | 600 | ok |

Notes: `audit agree` covers local answers the lead audited and forwarded items the understudent would have answered (eligible). hunch fields that are approximations are listed in the docstring of bench/hunch_runner/run_hunch.py.
