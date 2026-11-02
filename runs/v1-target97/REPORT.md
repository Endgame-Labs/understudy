# understudy-bench run `v1-target97`


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
| world_knowledge | capital_city | noul | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.3% | 73.0% | 0/600 | 31266 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 72.8% | 0/600 | 21111 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 73.4% | 0/600 | 34355 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 69.8% | 0/600 | 38 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 57.4% | 0/600 | 93 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 72.0% | 0/600 | 21276 | **correctly abstains** | abstains |
| world_knowledge | capital_city | noul | understudent-ladder | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.2% | 99.2% | 73.4% | 0/600 | 35090 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 45.0% | 0/600 | 67440 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 45.0% | 0/600 | 52504 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 56.4% | 0/600 | 69555 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 25.0% | 0/600 | 72 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 26.4% | 0/600 | 148 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 34.6% | 0/600 | 69840 | **correctly abstains** | abstains |
| binding_lookup | code_lookup | choice | understudent-ladder | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.5% | 99.5% | 56.4% | 0/600 | 74016 | **correctly abstains** | abstains |
| graded_tone | customer_frustration | score | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 54.4% | 0/600 | 91418 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 54.4% | 0/600 | 75948 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 56.8% | 0/600 | 106803 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 49.8% | 0/600 | 210 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 50.8% | 0/600 | 504 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 47.6% | 0/600 | 78935 | **correctly abstains** | partial |
| graded_tone | customer_frustration | score | understudent-ladder | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 65.5% | 65.5% | 56.8% | 0/600 | 164502 | **correctly abstains** | partial |
| compound_judgment | incident_severity | score | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 51.6% | 0/600 | 135458 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 51.6% | 0/600 | 105174 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 54.4% | 0/600 | 229779 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 50.2% | 0/600 | 329 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 48.8% | 0/600 | 526 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 47.2% | 0/600 | 118155 | **correctly abstains** | abstains |
| compound_judgment | incident_severity | score | understudent-ladder | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 67.2% | 67.2% | 54.4% | 0/600 | 215845 | **correctly abstains** | abstains |
| topic_kind | output_kind | choice | hunch | yes | 401 | 22.2% | 44.3% | 100.0% (133) | 100.0% (10) | 26.3% | 1.00 | 100.0% (98) | 95.8% | 95.8% | 91.6% | 123/600 | 158257 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-probe | yes | 401 | 22.3% | 44.7% | 100.0% (134) | 100.0% (10) | 26.1% | 1.00 | 100.0% (99) | 95.8% | 95.8% | 91.6% | 124/600 | 127939 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-deep | yes | 401 | 27.0% | 54.0% | 98.8% (162) | 100.0% (12) | 24.1% | 0.98 | 99.2% (123) | 95.8% | 95.8% | 94.0% | 150/600 | 329205 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-keywords | yes | 403 | 19.8% | 39.7% | 98.3% (119) | 100.0% (7) | 28.6% | 0.98 | 97.6% (85) | 95.8% | 96.2% | 90.4% | 112/600 | 207 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-lite | yes | 301 | 22.0% | 44.0% | 98.5% (132) | 100.0% (9) | 28.8% | 0.98 | 98.9% (94) | 95.8% | 96.2% | 89.0% | 123/600 | 557 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-neighbors | yes | 501 | 9.7% | 19.3% | 100.0% (58) | 100.0% (3) | 27.6% | 1.00 | 100.0% (42) | 95.8% | 95.8% | 82.4% | 55/600 | 226001 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-ladder | yes | 401 | 28.8% | 57.7% | 97.7% (173) | 92.9% (14) | 23.7% | 0.97 | 97.7% (132) | 95.8% | 96.2% | 94.0% | 159/600 | 288603 | **takes over** | takes over |
| relational_match | passage_answers | noul | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.8% | 75.0% | 0/600 | 254501 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 80.2% | 0/600 | 214665 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 82.6% | 0/600 | 311107 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 75.4% | 0/600 | 366 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 75.8% | 0/600 | 1583 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 67.4% | 0/600 | 198133 | **correctly abstains** | abstains |
| relational_match | passage_answers | noul | understudent-ladder | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 93.7% | 93.7% | 82.6% | 0/600 | 289245 | **correctly abstains** | abstains |
| minimal_pair | payment_flip | noul | hunch | yes | 402 | 20.7% | 41.3% | 98.4% (124) | 100.0% (7) | 71.0% | 0.96 | 97.2% (36) | 99.7% | 99.3% | 87.0% | 117/600 | 99442 | **partial** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-probe | yes | 402 | 20.8% | 41.7% | 98.4% (125) | 100.0% (8) | 71.2% | 0.96 | 97.2% (36) | 99.7% | 99.3% | 86.8% | 117/600 | 86841 | **partial** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-deep | yes | 301 | 34.3% | 68.7% | 100.0% (206) | 100.0% (16) | 75.7% | 1.00 | 100.0% (50) | 99.7% | 99.7% | 93.2% | 190/600 | 173154 | **takes over** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 92.0% | 0/600 | 154 | **correctly abstains** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 85.4% | 0/600 | 252 | **correctly abstains** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 72.2% | 0/600 | 156757 | **correctly abstains** | partial or abstains |
| minimal_pair | payment_flip | noul | understudent-ladder | yes | 301 | 34.3% | 68.7% | 100.0% (206) | 100.0% (16) | 75.7% | 1.00 | 100.0% (50) | 99.7% | 99.7% | 93.2% | 190/600 | 228199 | **takes over** | partial or abstains |
| pair_compare | politer_reply | choice | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.8% | 99.8% | 80.0% | 0/600 | 358987 | **correctly abstains** | abstains |
| pair_compare | politer_reply | choice | understudent-probe | yes | 201 | 63.5% | 95.7% | 100.0% (381) | 100.0% (40) | 53.0% | 1.00 | 100.0% (179) | 99.8% | 99.8% | 98.8% | 341/600 | 277774 | **takes over** | ? |
| pair_compare | politer_reply | choice | understudent-deep | yes | 201 | 63.5% | 95.7% | 100.0% (381) | 100.0% (40) | 53.0% | 1.00 | 100.0% (179) | 99.8% | 99.8% | 98.4% | 341/600 | 382196 | **takes over** | ? |
| pair_compare | politer_reply | choice | understudent-keywords | yes | 301 | 28.8% | 57.7% | 100.0% (173) | 100.0% (14) | 53.8% | 1.00 | 100.0% (80) | 99.8% | 99.8% | 95.0% | 159/600 | 454 | **takes over** | ? |
| pair_compare | politer_reply | choice | understudent-lite | yes | 301 | 45.0% | 90.0% | 100.0% (270) | 100.0% (24) | 54.8% | 1.00 | 100.0% (122) | 99.8% | 99.8% | 97.6% | 246/600 | 1569 | **takes over** | ? |
| pair_compare | politer_reply | choice | understudent-neighbors | yes | 301 | 44.3% | 88.7% | 99.6% (266) | 100.0% (23) | 52.3% | 0.99 | 100.0% (127) | 99.8% | 99.7% | 96.6% | 243/600 | 238966 | **takes over** | ? |
| pair_compare | politer_reply | choice | understudent-ladder | yes | 201 | 63.5% | 95.7% | 100.0% (381) | 100.0% (40) | 53.0% | 1.00 | 100.0% (179) | 99.8% | 99.8% | 98.4% | 341/600 | 366248 | **takes over** | ? |
| conditional_rule | retry_exception | noul | hunch | yes | 204 | 32.7% | 41.7% | 95.4% (196) | 100.0% (15) | 79.1% | 0.85 | 78.0% (41) | 100.0% | 98.5% | 89.6% | 181/600 | 835954 | **partial** | abstains |
| conditional_rule | retry_exception | noul | understudent-probe | yes | 204 | 32.7% | 41.7% | 95.4% (196) | 100.0% (15) | 79.1% | 0.85 | 78.0% (41) | 100.0% | 98.5% | 89.6% | 181/600 | 805895 | **partial** | abstains |
| conditional_rule | retry_exception | noul | understudent-deep | yes | 401 | 26.0% | 52.0% | 98.1% (156) | 100.0% (12) | 77.6% | 0.94 | 91.4% (35) | 100.0% | 99.5% | 92.4% | 144/600 | 1096814 | **takes over** | abstains |
| conditional_rule | retry_exception | noul | understudent-keywords | yes | 402 | 16.5% | 33.0% | 99.0% (99) | 80.0% (5) | 98.0% | 0.66 | 50.0% (2) | 100.0% | 99.8% | 88.6% | 94/600 | 601 | **partial** | abstains |
| conditional_rule | retry_exception | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 83.6% | 0/600 | 1430 | **correctly abstains** | abstains |
| conditional_rule | retry_exception | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 100.0% | 100.0% | 74.0% | 0/600 | 916936 | **correctly abstains** | abstains |
| conditional_rule | retry_exception | noul | understudent-ladder | yes | 401 | 26.8% | 53.7% | 98.1% (161) | 100.0% (12) | 78.3% | 0.94 | 91.4% (35) | 100.0% | 99.5% | 92.6% | 149/600 | 1123973 | **takes over** | abstains |
| needle | secret_needle | noul | hunch | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 68.0% | 0/600 | 734387 | **correctly abstains** | abstains or partial |
| needle | secret_needle | noul | understudent-probe | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 67.8% | 0/600 | 779717 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 72.4% | 0/600 | 1775061 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-keywords | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 71.0% | 0/600 | 1217 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 64.4% | 0/600 | 3266 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 63.0% | 0/600 | 1162210 | **correctly abstains** | ? |
| needle | secret_needle | noul | understudent-ladder | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 99.7% | 99.7% | 72.4% | 0/600 | 2103902 | **correctly abstains** | ? |
| surface_property | tool_failed | noul | hunch | yes | 302 | 39.2% | 78.3% | 97.4% (235) | 94.7% (19) | 78.7% | 0.92 | 92.0% (50) | 92.3% | 92.7% | 90.0% | 216/600 | 342270 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-probe | yes | 302 | 39.2% | 78.3% | 97.4% (235) | 94.7% (19) | 78.7% | 0.92 | 92.0% (50) | 92.3% | 92.7% | 90.0% | 216/600 | 351693 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-deep | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 92.3% | 92.3% | 89.6% | 0/600 | 496351 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-keywords | yes | 302 | 32.3% | 64.7% | 98.5% (194) | 100.0% (15) | 94.8% | 0.82 | 70.0% (10) | 92.3% | 92.8% | 81.8% | 179/600 | 203 | **takes over** | takes over |
| surface_property | tool_failed | noul | understudent-lite | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 92.3% | 92.3% | 82.6% | 0/600 | 455 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-neighbors | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 92.3% | 92.3% | 83.4% | 0/600 | 338198 | **correctly abstains** | takes over |
| surface_property | tool_failed | noul | understudent-ladder | yes | 302 | 32.3% | 64.7% | 98.5% (194) | 100.0% (15) | 94.8% | 0.82 | 70.0% (10) | 92.3% | 92.8% | 85.8% | 179/600 | 305423 | **takes over** | takes over |

## Winners by category (cold start)

Winner = a system that graduates with accuracy within 1 pt of the lead alone, ranked by 2nd-half coverage, ties to the lower µs/item.

| category | dataset | type | winner | cov. 2nd half | acc system | acc lead | µs/item | runner-up |
|---|---|---|---|---|---|---|---|---|
| world_knowledge | capital_city | noul | none | | | | | |
| binding_lookup | code_lookup | choice | none | | | | | |
| graded_tone | customer_frustration | score | none | | | | | |
| compound_judgment | incident_severity | score | none | | | | | |
| topic_kind | output_kind | choice | **understudent-ladder** | 57.7% | 96.2% | 95.8% | 288603 | understudent-deep (54.0%) |
| relational_match | passage_answers | noul | none | | | | | |
| minimal_pair | payment_flip | noul | **understudent-deep** | 68.7% | 99.7% | 99.7% | 173154 | understudent-ladder (68.7%) |
| pair_compare | politer_reply | choice | **understudent-probe** | 95.7% | 99.8% | 99.8% | 277774 | understudent-ladder (95.7%) |
| conditional_rule | retry_exception | noul | **understudent-ladder** | 53.7% | 99.5% | 100.0% | 1123973 | understudent-deep (52.0%) |
| needle | secret_needle | noul | none | | | | | |
| surface_property | tool_failed | noul | **hunch** | 78.3% | 92.7% | 92.3% | 342270 | understudent-probe (78.3%) |

## Coverage by readout (from each system's summary)

| dataset | system | coverage_by_readout |
|---|---|---|
| output_kind | understudent-probe | mean 22.3% |
| output_kind | understudent-deep | pool 27.0% |
| output_kind | understudent-keywords | keywords 19.8% |
| output_kind | understudent-lite | lite 22.0% |
| output_kind | understudent-neighbors | neighbors 9.7% |
| output_kind | understudent-ladder | keywords 19.8%, deep:pool 9.0% |
| payment_flip | understudent-probe | mean 20.8% |
| payment_flip | understudent-deep | pool 34.3% |
| payment_flip | understudent-ladder | deep:pool 34.3% |
| politer_reply | understudent-probe | mean 63.5% |
| politer_reply | understudent-deep | attn 63.5% |
| politer_reply | understudent-keywords | keywords 28.8% |
| politer_reply | understudent-lite | lite 45.0% |
| politer_reply | understudent-neighbors | neighbors 44.3% |
| politer_reply | understudent-ladder | deep:attn 63.5% |
| retry_exception | understudent-probe | mean 32.7% |
| retry_exception | understudent-deep | pool 26.0% |
| retry_exception | understudent-keywords | keywords 16.5% |
| retry_exception | understudent-ladder | keywords 15.3%, deep:pool 11.5% |
| tool_failed | understudent-probe | mean 39.2% |
| tool_failed | understudent-keywords | keywords 32.3% |
| tool_failed | understudent-ladder | keywords 32.3% |

## Summary by wire type

| type | system | datasets | takes over | partial | correctly abstains | harmful | mean cov. 2nd half | mean Δacc vs lead |
|---|---|---|---|---|---|---|---|---|
| choice | understudent-ladder | 3 | 2 | 0 | 1 | 0 | 51.1% | +0.1 pt |
| choice | understudent-neighbors | 3 | 1 | 1 | 1 | 0 | 36.0% | -0.1 pt |
| choice | understudent-keywords | 3 | 1 | 1 | 1 | 0 | 32.4% | +0.1 pt |
| choice | understudent-probe | 3 | 1 | 1 | 1 | 0 | 46.8% | +0.0 pt |
| choice | understudent-deep | 3 | 2 | 0 | 1 | 0 | 49.9% | +0.0 pt |
| choice | understudent-lite | 3 | 1 | 1 | 1 | 0 | 44.7% | +0.1 pt |
| choice | hunch | 3 | 0 | 1 | 2 | 0 | 14.8% | +0.0 pt |
| noul | understudent-ladder | 6 | 3 | 0 | 3 | 0 | 31.2% | +0.0 pt |
| noul | understudent-neighbors | 6 | 0 | 0 | 6 | 0 | 0.0% | +0.0 pt |
| noul | understudent-keywords | 6 | 1 | 1 | 4 | 0 | 16.3% | +0.1 pt |
| noul | understudent-probe | 6 | 1 | 2 | 3 | 0 | 26.9% | -0.3 pt |
| noul | understudent-deep | 6 | 2 | 0 | 4 | 0 | 20.1% | -0.1 pt |
| noul | understudent-lite | 6 | 0 | 0 | 6 | 0 | 0.0% | +0.0 pt |
| noul | hunch | 6 | 1 | 2 | 3 | 0 | 26.9% | -0.2 pt |
| score | understudent-ladder | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-neighbors | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-keywords | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-probe | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-deep | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | understudent-lite | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |
| score | hunch | 2 | 0 | 0 | 2 | 0 | 0.0% | +0.0 pt |

## Warm start vs cold start (operate items only: stream items 201..end)

Warm start = seeded with 200 lead answers via train, then judging. Cold start = the replay's decisions on the same items.

| dataset | category | system | graduated right after training | coverage | cov. 2nd half | acc lead | acc system | kappa | verdict |
|---|---|---|---|---|---|---|---|---|---|

## Timing

| dataset | system | wall s | train s | items | status |
|---|---|---|---|---|---|
| capital_city | hunch | 34.3 | – | 600 | ok |
| capital_city | understudent-probe | 14.8 | – | 600 | ok |
| capital_city | understudent-deep | 27.6 | – | 600 | ok |
| capital_city | understudent-keywords | 0.2 | – | 600 | ok |
| capital_city | understudent-lite | 2.2 | – | 600 | ok |
| capital_city | understudent-neighbors | 15.1 | – | 600 | ok |
| capital_city | understudent-ladder | 27.7 | – | 600 | ok |
| code_lookup | hunch | 64.4 | – | 600 | ok |
| code_lookup | understudent-probe | 33.5 | – | 600 | ok |
| code_lookup | understudent-deep | 54.5 | – | 600 | ok |
| code_lookup | understudent-keywords | 0.8 | – | 600 | ok |
| code_lookup | understudent-lite | 2.2 | – | 600 | ok |
| code_lookup | understudent-neighbors | 44.4 | – | 600 | ok |
| code_lookup | understudent-ladder | 58.7 | – | 600 | ok |
| customer_frustration | hunch | 75.9 | – | 600 | ok |
| customer_frustration | understudent-probe | 48.1 | – | 600 | ok |
| customer_frustration | understudent-deep | 83.6 | – | 600 | ok |
| customer_frustration | understudent-keywords | 3.9 | – | 600 | ok |
| customer_frustration | understudent-lite | 2.8 | – | 600 | ok |
| customer_frustration | understudent-neighbors | 49.3 | – | 600 | ok |
| customer_frustration | understudent-ladder | 125.8 | – | 600 | ok |
| incident_severity | hunch | 97.8 | – | 600 | ok |
| incident_severity | understudent-probe | 65.2 | – | 600 | ok |
| incident_severity | understudent-deep | 168.8 | – | 600 | ok |
| incident_severity | understudent-keywords | 5.5 | – | 600 | ok |
| incident_severity | understudent-lite | 2.5 | – | 600 | ok |
| incident_severity | understudent-neighbors | 72.8 | – | 600 | ok |
| incident_severity | understudent-ladder | 164.6 | – | 600 | ok |
| output_kind | hunch | 105.2 | – | 600 | ok |
| output_kind | understudent-probe | 79.0 | – | 600 | ok |
| output_kind | understudent-deep | 215.7 | – | 600 | ok |
| output_kind | understudent-keywords | 2.0 | – | 600 | ok |
| output_kind | understudent-lite | 2.7 | – | 600 | ok |
| output_kind | understudent-neighbors | 138.2 | – | 600 | ok |
| output_kind | understudent-ladder | 195.2 | – | 600 | ok |
| passage_answers | hunch | 168.8 | – | 600 | ok |
| passage_answers | understudent-probe | 131.4 | – | 600 | ok |
| passage_answers | understudent-deep | 219.9 | – | 600 | ok |
| passage_answers | understudent-keywords | 2.1 | – | 600 | ok |
| passage_answers | understudent-lite | 3.7 | – | 600 | ok |
| passage_answers | understudent-neighbors | 121.6 | – | 600 | ok |
| passage_answers | understudent-ladder | 205.5 | – | 600 | ok |
| payment_flip | hunch | 70.8 | – | 600 | ok |
| payment_flip | understudent-probe | 54.5 | – | 600 | ok |
| payment_flip | understudent-deep | 112.3 | – | 600 | ok |
| payment_flip | understudent-keywords | 0.8 | – | 600 | ok |
| payment_flip | understudent-lite | 3.0 | – | 600 | ok |
| payment_flip | understudent-neighbors | 96.9 | – | 600 | ok |
| payment_flip | understudent-ladder | 146.1 | – | 600 | ok |
| politer_reply | hunch | 235.0 | – | 600 | ok |
| politer_reply | understudent-probe | 169.8 | – | 600 | ok |
| politer_reply | understudent-deep | 239.3 | – | 600 | ok |
| politer_reply | understudent-keywords | 1.0 | – | 600 | ok |
| politer_reply | understudent-lite | 3.8 | – | 600 | ok |
| politer_reply | understudent-neighbors | 146.0 | – | 600 | ok |
| politer_reply | understudent-ladder | 228.0 | – | 600 | ok |
| retry_exception | hunch | 516.2 | – | 600 | ok |
| retry_exception | understudent-probe | 486.5 | – | 600 | ok |
| retry_exception | understudent-deep | 704.8 | – | 600 | ok |
| retry_exception | understudent-keywords | 1.6 | – | 600 | ok |
| retry_exception | understudent-lite | 3.7 | – | 600 | ok |
| retry_exception | understudent-neighbors | 553.0 | – | 600 | ok |
| retry_exception | understudent-ladder | 734.0 | – | 600 | ok |
| secret_needle | hunch | 455.3 | – | 600 | ok |
| secret_needle | understudent-probe | 469.9 | – | 600 | ok |
| secret_needle | understudent-deep | 1164.8 | – | 600 | ok |
| secret_needle | understudent-keywords | 5.9 | – | 600 | ok |
| secret_needle | understudent-lite | 5.7 | – | 600 | ok |
| secret_needle | understudent-neighbors | 699.9 | – | 600 | ok |
| secret_needle | understudent-ladder | 1369.3 | – | 600 | ok |
| tool_failed | hunch | 219.5 | – | 600 | ok |
| tool_failed | understudent-probe | 214.0 | – | 600 | ok |
| tool_failed | understudent-deep | 329.9 | – | 600 | ok |
| tool_failed | understudent-keywords | 0.6 | – | 600 | ok |
| tool_failed | understudent-lite | 3.0 | – | 600 | ok |
| tool_failed | understudent-neighbors | 205.9 | – | 600 | ok |
| tool_failed | understudent-ladder | 205.7 | – | 600 | ok |

Notes: `audit agree` covers local answers the lead audited and forwarded items the understudent would have answered (eligible). hunch fields that are approximations are listed in the docstring of bench/hunch_runner/run_hunch.py.
