# Datasets

Eleven small judgment tasks, 600 items each, used by `understudy-bench` (see [../docs/benchmarks.md](../docs/benchmarks.md)).

## Source and license

Every dataset here is **synthetic**. Each one was generated in this repo by `understudy-bench synth`, which asks an
open-weights model over an OpenAI-compatible chat API to write items to a per-item specification (the label is part of
the specification, so the truth is known by construction), then filters and orders them with a fixed seed. The generator
model, as recorded in each `synth_stats.json` (`model`), is `accounts/fireworks/models/glm-5p3-flash` (GLM 5.3 Flash).
`code_lookup` is built programmatically with no model call, so its `synth_stats.json` has `"programmatic": true` and no
`model`.

The generated text is synthetic: no real people, credentials, customers or private records. Names, numbers and secrets
in the items are invented. These files are released under this repo's MIT license (see [../LICENSE](../LICENSE)).
Nothing in `data/` comes from a third-party corpus.

## Files per dataset

| File | What it is |
|---|---|
| `question.json` | the question, its answer options and the category |
| `items.jsonl` | the items: `id`, `state` (the text being judged) and `truth` (the label the item was generated with) |
| `lead.<name>.jsonl` | recorded answers from an example lead model (probabilities per option, latency, model id), one line per item. The benchmark replays these instead of calling a lead, so runs are reproducible. Two leads are included, `jev` and `rune` (see [Lead answers](#lead-answers)). |
| `synth_stats.json` | how the dataset was generated: size, class counts, seed, dropped batches, generator model |

## Datasets

| Dataset | Category | Question |
|---|---|---|
| `capital_city` | world knowledge | Is the named city the capital of the named country? |
| `code_lookup` | binding lookup | Which candidate's code equals the target code? (programmatic) |
| `customer_frustration` | graded tone | How frustrated is the customer who wrote this support message? (0 to 3) |
| `incident_severity` | compound judgment | Rate an incident's severity from its impact, scope and resolution (0 to 3) |
| `output_kind` | topic kind | What kind of output did this coding-agent tool call return? |
| `passage_answers` | relational match | Does the passage answer the query? (both the query and the passage change per item) |
| `payment_flip` | minimal pair | Did the payment succeed? |
| `politer_reply` | pair compare | Which of two support replies is more polite? |
| `retry_exception` | conditional rule | Did the deploy fail (a step that failed and then succeeded on retry does not count)? |
| `secret_needle` | needle | Does this log contain a credential or secret anywhere? |
| `tool_failed` | surface property | Did this coding-agent tool call fail? |

## Regenerating

`understudy-bench synth` needs an API key for the generator's endpoint. Regenerating produces different items and
invalidates the committed `lead.*.jsonl` files and runs, which are tied to these exact items by `state_sha`.

## Lead answers

Every `lead.*.jsonl` answers only the synthetic items in this directory; no answer was recorded for any other data.

| file | lead model | model id recorded | when |
|---|---|---|---|
| `lead.jev.jsonl` | Jev (TypeSafe), via its System One API | `jev-1.14.0` | 2026-11-01 |
| `lead.rune.jsonl` | Rune (Fireworks), via its System One API | `accounts/fireworks/routers/rune` (a router; no version is reported) | 2026-10-29 to 2026-11-01 |
