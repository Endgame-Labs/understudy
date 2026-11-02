# understudy-bench run `latencyd-jev`


Systems: `understudent` / `hunch` = cold start (replay of the whole stream from an empty store); `*-split` = warm start (seeded with 200 lead answers via train, then judging the remaining items; its positions, halves, accuracies and calls are over those operate items only, not counting the 200 seed answers).

Verdicts: **takes over** = 2nd-half coverage ≥ 50% and accuracy within 1 pt of the lead alone; **harmful** = graduated and accuracy > 2 pts below the lead; **correctly abstains** = never graduated; otherwise **partial**. Accuracy is against the generator's labels (ground truth by construction). Agreement, kappa and minority recall are against the lead, on the items the understudent answered.

## Lead vs ground truth

| dataset | category | type | n | class counts | lead acc | recall per class | flag |
|---|---|---|---|---|---|---|---|
| politer_reply | pair_compare | choice | 600 | [300 300] | 99.8% | 1.00 / 1.00 |  |
| output_kind | topic_kind | choice | 600 | [120 120 120 120 120] | 96.3% | 0.98 / 1.00 / 0.86 / 1.00 / 0.97 |  |

## Results by category

| category | dataset | type | system | grad. | first local | coverage | cov. 2nd half | local agree | audit agree | majority base | kappa | minority recall | acc lead | acc system | acc understudent | calls saved | µs/item | verdict | expected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pair_compare | politer_reply | choice | understudent-keywords | yes | 101 | 80.0% | 97.7% | 94.2% (480) | 90.0% (50) | 47.9% | 0.88 | 91.2% (250) | 99.8% | 95.2% | 93.6% | 430/600 | 299 | **harmful** | ? |
| pair_compare | politer_reply | choice | understudent-shape | yes | 201 | 58.5% | 87.0% | 88.9% (351) | 83.7% (49) | 49.0% | 0.78 | 89.4% (179) | 99.8% | 93.3% | 87.6% | 317/600 | 77 | **harmful** | ? |
| pair_compare | politer_reply | choice | understudent-checklist | yes | 201 | 60.2% | 91.0% | 97.5% (361) | 97.1% (35) | 47.9% | 0.95 | 98.9% (188) | 99.8% | 98.3% | 95.2% | 326/600 | 234 | **partial** | ? |
| pair_compare | politer_reply | choice | understudent-lite | yes | 101 | 73.2% | 90.3% | 97.5% (439) | 95.7% (47) | 49.2% | 0.95 | 96.4% (223) | 99.8% | 98.0% | 96.0% | 392/600 | 236 | **partial** | ? |
| pair_compare | politer_reply | choice | understudent-deep | yes | 101 | 78.2% | 95.3% | 98.7% (469) | 100.0% (49) | 48.6% | 0.97 | 97.9% (241) | 99.8% | 98.8% | 97.8% | 420/600 | 86915 | **takes over** | ? |
| pair_compare | politer_reply | choice | understudent-auto | yes | 101 | 80.0% | 97.7% | 94.2% (480) | 90.0% (50) | 47.9% | 0.88 | 91.2% (250) | 99.8% | 95.2% | 93.6% | 430/600 | 17875 | **harmful** | ? |
| topic_kind | output_kind | choice | understudent-keywords | yes | 102 | 51.7% | 70.3% | 98.1% (310) | 100.0% (31) | 16.8% | 0.98 | 98.4% (258) | 96.3% | 95.3% | 86.2% | 279/600 | 139 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-shape | no | – | 0.0% | 0.0% | – (0) | – (0) | – | – | – (0) | 96.3% | 96.3% | 65.4% | 0/600 | 74 | **correctly abstains** | takes over |
| topic_kind | output_kind | choice | understudent-checklist | yes | 401 | 21.2% | 42.3% | 95.3% (127) | 88.9% (9) | 15.7% | 0.94 | 94.4% (107) | 96.3% | 95.3% | 89.2% | 118/600 | 161 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-lite | yes | 201 | 50.8% | 77.0% | 97.7% (305) | 100.0% (28) | 20.3% | 0.97 | 98.4% (243) | 96.3% | 95.2% | 89.4% | 277/600 | 355 | **partial** | takes over |
| topic_kind | output_kind | choice | understudent-deep | yes | 102 | 57.7% | 80.7% | 97.7% (346) | 94.1% (34) | 18.5% | 0.97 | 98.9% (282) | 96.3% | 96.3% | 92.6% | 312/600 | 100923 | **takes over** | takes over |
| topic_kind | output_kind | choice | understudent-auto | yes | 102 | 55.2% | 76.3% | 98.5% (331) | 100.0% (33) | 15.7% | 0.98 | 98.9% (279) | 96.3% | 95.5% | 88.6% | 298/600 | 108474 | **takes over** | takes over |

## Winners by category (cold start)

Winner = a system that graduates with accuracy within 1 pt of the lead alone, ranked by 2nd-half coverage, ties to the lower µs/item.

| category | dataset | type | winner | cov. 2nd half | acc system | acc lead | µs/item | runner-up |
|---|---|---|---|---|---|---|---|---|
| pair_compare | politer_reply | choice | **understudent-deep** | 95.3% | 98.8% | 99.8% | 86915 | – |
| topic_kind | output_kind | choice | **understudent-deep** | 80.7% | 96.3% | 96.3% | 100923 | understudent-auto (76.3%) |

## Coverage by readout (from each system's summary)

| dataset | system | coverage_by_readout |
|---|---|---|
| politer_reply | understudent-keywords | keywords 80.0% |
| politer_reply | understudent-shape | shape 58.5% |
| politer_reply | understudent-checklist | checklist 60.2% |
| politer_reply | understudent-lite | lite 73.2% |
| politer_reply | understudent-deep | pool 78.2% |
| politer_reply | understudent-auto | keywords 80.0% |
| output_kind | understudent-keywords | keywords 51.7% |
| output_kind | understudent-checklist | checklist 21.2% |
| output_kind | understudent-lite | lite 50.8% |
| output_kind | understudent-deep | pool 57.7% |
| output_kind | understudent-auto | keywords 14.8%, deep:pool 40.3% |

## Summary by wire type

| type | system | datasets | takes over | partial | correctly abstains | harmful | mean cov. 2nd half | mean Δacc vs lead |
|---|---|---|---|---|---|---|---|---|
| choice | understudent-auto | 2 | 1 | 0 | 0 | 1 | 87.0% | -2.8 pt |
| choice | understudent-checklist | 2 | 0 | 2 | 0 | 0 | 66.7% | -1.3 pt |
| choice | understudent-shape | 2 | 0 | 0 | 1 | 1 | 43.5% | -3.2 pt |
| choice | understudent-keywords | 2 | 1 | 0 | 0 | 1 | 84.0% | -2.8 pt |
| choice | understudent-deep | 2 | 2 | 0 | 0 | 0 | 88.0% | -0.5 pt |
| choice | understudent-lite | 2 | 0 | 2 | 0 | 0 | 83.7% | -1.5 pt |

## Warm start vs cold start (operate items only: stream items 201..end)

Warm start = seeded with 200 lead answers via train, then judging. Cold start = the replay's decisions on the same items.

| dataset | category | system | graduated right after training | coverage | cov. 2nd half | acc lead | acc system | kappa | verdict |
|---|---|---|---|---|---|---|---|---|---|

## Usefulness

Per dataset × system, over every item of the run. **coverage** = share answered by the understudent; **agree** = agreement with the lead on that share, with its Wilson lower bound (z = 1.96); **us** = the understudent's own time (`understudent_us`) p50/p99 on the items it answered, **fwd us** on the items it forwarded (the overhead before the lead); **lead ms** = the lead's recorded latency p50/p99 (`latency_ms`); **$ saved / 1k calls** = coverage × lead cost per call × 1000. Lead cost per call = (state bytes / 3.4 + 600) tokens × the rate of the lead (Jev and Rune: $0.042 per million uncached input tokens, output free; Fireworks billing API 2026-11-01; TypeSafe list price), averaged over the stream's states.

| dataset | system | coverage | agree (Wilson low) | us p50/p99 | fwd us p50/p99 | lead ms p50/p99 | lead $/call | $ saved / 1k calls |
|---|---|---|---|---|---|---|---|---|
| politer_reply | understudent-keywords | 80.0% | 94.2% (91.7%) n=480 | 208 / 4784 | 149 / 368 | 82.0 / 204.0 | $0.000033 | $0.026614 |
| politer_reply | understudent-shape | 58.5% | 88.9% (85.2%) n=351 | 74 / 154 | 70 / 251 | 82.0 / 204.0 | $0.000033 | $0.019461 |
| politer_reply | understudent-checklist | 60.2% | 97.5% (95.3%) n=361 | 236 / 728 | 189 / 519 | 82.0 / 204.0 | $0.000033 | $0.020016 |
| politer_reply | understudent-lite | 73.2% | 97.5% (95.6%) n=439 | 223 / 464 | 226 / 494 | 82.0 / 204.0 | $0.000033 | $0.024340 |
| politer_reply | understudent-deep | 78.2% | 98.7% (97.2%) n=469 | 88636 / 172124 | 72174 / 125808 | 82.0 / 204.0 | $0.000033 | $0.026004 |
| politer_reply | understudent-auto | 80.0% | 94.2% (91.7%) n=480 | 233 / 582 | 87717 / 174062 | 82.0 / 204.0 | $0.000033 | $0.026614 |
| output_kind | understudent-keywords | 51.7% | 98.1% (95.8%) n=310 | 103 / 537 | 74 / 549 | 78.0 / 152.0 | $0.000032 | $0.016292 |
| output_kind | understudent-shape | 0.0% | – | – / – | 60 / 218 | 78.0 / 152.0 | $0.000032 | $0.000000 |
| output_kind | understudent-checklist | 21.2% | 95.3% (90.1%) n=127 | 153 / 613 | 98 / 631 | 78.0 / 152.0 | $0.000032 | $0.006674 |
| output_kind | understudent-lite | 50.8% | 97.7% (95.3%) n=305 | 188 / 1626 | 197 / 4313 | 78.0 / 152.0 | $0.000032 | $0.016029 |
| output_kind | understudent-deep | 57.7% | 97.7% (95.5%) n=346 | 66532 / 476186 | 59899 / 533948 | 78.0 / 152.0 | $0.000032 | $0.018184 |
| output_kind | understudent-auto | 55.2% | 98.5% (96.5%) n=331 | 44141 / 536808 | 74822 / 714414 | 78.0 / 152.0 | $0.000032 | $0.017396 |

## Timing

| dataset | system | wall s | train s | items | status |
|---|---|---|---|---|---|
| politer_reply | understudent-keywords | 0.2 | – | 600 | ok |
| politer_reply | understudent-shape | 0.1 | – | 600 | ok |
| politer_reply | understudent-checklist | 0.2 | – | 600 | ok |
| politer_reply | understudent-lite | 1.3 | – | 600 | ok |
| politer_reply | understudent-deep | 54.3 | – | 600 | ok |
| politer_reply | understudent-auto | 17.3 | – | 600 | ok |
| output_kind | understudent-keywords | 1.0 | – | 600 | ok |
| output_kind | understudent-shape | 0.2 | – | 600 | ok |
| output_kind | understudent-checklist | 0.3 | – | 600 | ok |
| output_kind | understudent-lite | 1.6 | – | 600 | ok |
| output_kind | understudent-deep | 64.4 | – | 600 | ok |
| output_kind | understudent-auto | 75.7 | – | 600 | ok |

Notes: `audit agree` covers local answers the lead audited and forwarded items the understudent would have answered (eligible). hunch fields that are approximations are listed in the docstring of bench/hunch_runner/run_hunch.py.
