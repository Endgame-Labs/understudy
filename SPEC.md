# understudy: reference spec

The precise reference for Understudy: statistics, wire types, files, CLI and snapshot format. For explanations and diagrams
start with [README.md](README.md) and [docs/](docs/): [how it works](docs/how-it-works.md),
[graduation](docs/graduation.md), [the understudent designs](docs/understudents/README.md), [the API](docs/api.md),
[examples](docs/examples.md) and [benchmarks](docs/benchmarks.md).

Understudy is a CPU-only layer in front of a lead decision model (for example Jev or Rune: anything speaking the
`POST /v1/systemone` protocol). Per question it learns from the lead's answers, graduates on cross-validated Wilson
bounds, forwards novel inputs, audits a sample forever and demotes on bad audits. Several understudent designs (features plus
a model) run under the identical routing and statistics.

## Layout

```
Cargo.toml            understudy (lib + bin), depends on vendor/eg2cpu
src/                  Rust: the proxy, understudents, readouts, statistics, leads, CLI
vendor/eg2cpu/        EmbeddingGemma 2 CPU engine, patched for per-token states (see its PATCHED.md)
bench/                Go/Cobra harness `understudy-bench`: synth, teach, run, report
bench/hunch_runner/   uv project: replays a stream through the external `hunch` baseline proxy (optional)
data/<dataset>/       question.json, items.jsonl, lead.<lead>.jsonl, synth_raw.jsonl
runs/<run>/<dataset>/ stream.jsonl, decisions.<system>.jsonl, timing.<system>.json
```

### Model lookup and fetch

The checkpoint is `google/embeddinggemma-2` at revision `914f7f89142e33e77833254d9c9b90c3cef7303b` (Apache-2.0, not
gated). The loader reads three files: `config.json`, `tokenizer.json` and `model.safetensors`.
`eg2cpu::default_model_dir()` resolves the directory in this order:

1. `$EG2_MODEL_DIR`, used as given and **never hash-checked** (loading stays fast).
2. In the hub cache (`$HF_HUB_CACHE`, else `$HF_HOME/hub`, else `~/.cache/huggingface/hub`), the pinned revision's
   snapshot `models--google--embeddinggemma-2/snapshots/<revision>/`, when it holds all three files.
3. Otherwise the newest other snapshot that holds all three (by the mtime of `model.safetensors`), with a one-line
   warning on stderr naming the revision it used. Never `read_dir` order.
4. Otherwise the error `no EmbeddingGemma 2 model found (looked in <dir>): run `understudy model fetch`, or set
   EG2_MODEL_DIR. Designs that need no model: keywords, shape, memo, checklist.`

Pinned SHA-256 (in `src/modelfetch.rs`; the two LFS files equal the hub's `x-linked-etag`):

| file | bytes | sha256 |
|---|---|---|
| `config.json` | 4455 | `b8f1e993…1ad5bb3` |
| `tokenizer.json` | 32170510 | `4d777ef5…4828be4` |
| `model.safetensors` | 1488915288 | `197a3296…5553d79` |

## Wire types

Leads speak the systemone protocol (or, through the chat adapter below, an OpenAI-compatible chat API). Request: `{"model", "questions": {name: question}, "state": any JSON}`.
Question: `{"type": "noul"|"choice"|"score", "instructions": str, "criteria": ...}`:
- `noul`: criteria `{"true": str, "false": str}`; answer `{"type":"noul","noul": p_yes}`.
- `choice`: criteria `{option: description, ...}`; answer `{"type":"choice","choice": opt, "probabilities": {opt: p}, "confidence"}`.
- `score`: criteria an **ordered array** of level descriptions; answer
  `{"type":"score","score": expected, "probabilities": {"0": p, "1": p, ...}, "confidence"}`.

There is no `bool` type on the wire: a "bool" decision is a `noul` question whose ground truth is a hard yes/no.

**Chat leads.** A lead of kind `chat` is an OpenAI-compatible `POST <url>/chat/completions` endpoint. Everything inside Understudy
still speaks System One: the adapter (`src/chat.rs`) turns each question of a request into one chat call and the reply into the
answer shapes above, so the snapshot, the statistics and `serve` are unchanged.
- **One call per question**, concurrently (not one per request): each call has one small answer schema, a failure costs one
  question, and the answer token's logprobs belong to that question alone. The state is repeated per question.
- **Prompt** (exact text in `docs/api.md`): a fixed system message and a user message with the question's instructions, its
  criteria, the answer format and the state. The reply is asked as `{"answer": ...}`: a JSON boolean (`noul`), one of the
  criteria keys (`choice`) or a level index (`score`). `temperature` 0; `max_tokens` 2048.
- **`response_format`** is a `json_schema` (`{"answer"}` with a boolean, an enum of the options, or an enum of level indices). If
  the endpoint answers 400 or 422 the call is repeated once without `response_format` and `logprobs`, remembered for later calls,
  and the JSON is parsed out of the text (the whole text, a fenced block, or the first balanced `{...}`). A reply with no readable
  answer, or one outside the question's options, is an error (a 502 in `serve`); nothing is guessed.
- **Probabilities.** A chat API returns no class probabilities. With `logprobs` (`top_logprobs` 5), the answer token's alternatives
  that identify exactly one option by prefix (case-insensitive) give that option's mass; the masses are renormalized, floored at
  1e-4, and used only if they hold at least half the probability and their argmax is the option the text chose. Otherwise
  (no logprobs, ambiguous tokens, a disagreeing argmax) the chosen option gets the **fixed confidence 0.9**
  (`chat::DEFAULT_CONFIDENCE`, a stand-in, not a calibrated estimate; `confidence` in the preset changes it, clamped to
  [0.55, 1], so the chosen class always stays the argmax) and the others share the rest equally. Those answers are then built exactly like a System One lead's
  (`noul`: `p_yes`; `choice`: probabilities and confidence; `score`: expected level). The response says which, per
  question: `"understudy": {"adapter": "chat", "probabilities": {name: "logprobs"|"fixed"}}`. Soft labels from a fixed confidence
  make the learners' lead-probability targets coarser than a native lead's; keep that in mind when comparing leads.
- **Errors.** A non-2xx status from the chat endpoint is relayed as it is (status, body, `Content-Type` and `Retry-After`) by
  `serve`, and retried (408, 429, 5xx, transport) by the CLI leads like System One ones. If the schema request is refused (400/422)
  and the plain retry fails too, the retry's failure is the one reported (a 429 there stays retryable).
- **Limits.** At most 8 questions of a request are in flight at once. A response (or error body) over the limit (16 MiB for the
  CLI leads; `--max-response-mib` in `serve`, where it is a 502 `upstream_response_too_large`) is never read further or truncated.

**Class indices** are used everywhere in files: `noul` → 0 = false/no, 1 = true/yes; `choice` → index into the
`options` list in question.json (explicit, because JSON object order is not reliable in Go); `score` → level index.

## Files

`data/<ds>/question.json`
```json
{"name": "tool_failed", "category": "surface_property", "type": "noul",
 "options": ["false", "true"],
 "question": {"type": "noul", "instructions": "...", "criteria": {"true": "...", "false": "..."}},
 "pair": null,
 "notes": "what this dataset probes, and the predicted outcome"}
```
`pair` is `null` or `["a", "b"]`: two top-level state fields that understudent may embed separately (pair features).
The `hunch` baseline ignores it (it reads the whole state as one text).

`data/<ds>/items.jsonl`: `{"id": "tool_failed-0001", "state": {...}, "truth": 1}` in stream order.

`data/<ds>/lead.<lead>.jsonl`: `{"id", "probs": [p_0, ..., p_k-1], "label": argmax, "answer": {raw answer object},
"latency_ms": n, "model": "..."}`. For `noul`, `probs = [1 - p_yes, p_yes]`.

`runs/<run>/<ds>/stream.jsonl` (the runners' only input besides question.json; **no truth in it**):
`{"id", "state", "lead_probs": [...]}`.

`runs/<run>/<ds>/decisions.<system>.jsonl`, one line per stream item, in order:
```json
{"id": "...", "answered_by": "understudent"|"lead",
 "label": 1, "probs": [...],                 // what the caller got
 "understudent_label": 0, "understudent_probs": [...], // the understudent's own prediction, null before any fit
 "eligible": false,                           // the understudent would have answered (graduated, confident, not novel)
 "novel": false, "demoted": false, "mode": "symmetric"|"one_sided"|"one_sided_yes"|null,
 "audited": false, "readout": "pool"|"attn"|"mean"|"lite"|"neighbors"|"keywords"|"shape"|"blend"|"memo"|"rules"|"deep:pool"|"deep:attn"|null,   // a ladder: the answering tier
 "understudent_us": 1234}                          // wall time of encode+predict for this item, microseconds
```

## Replay semantics (both runners, deterministic)

Items are processed one at a time in stream order. For each item: encode; predict with the current fit; if the
understudent may answer (graduated, confident per its mode, not novel, not demoted) it answers, and with probability
`audit_rate` (seeded RNG, seed 0) the lead's answer is also used as an audit and a new example. Otherwise the
item is forwarded: the caller gets `lead_probs`, the example is stored, and every understudent that *would* have
answered records an audit. A refit runs **synchronously** whenever a question has 100 new examples since the last
fit (the proxies normally refit in the background; replay waits so runs are reproducible).

## Statistics

Explained with diagrams and worked numbers in [docs/graduation.md](docs/graduation.md).

- Examples: rotating window of the 2,000 most recent lead answers per question; one example per state hash
  (a new answer replaces the old one); label = argmax of lead probs; weight = lead confidence (max prob).
- Fit: 5-fold CV (shuffled, seed 0) for out-of-fold probabilities; final model on all examples.
- **Symmetric graduation**: lowest confidence threshold t in 0.50, 0.51, … 0.99 such that among OOF predictions
  with confidence ≥ t: ≥ 30 of them, Wilson lower bound of agreement ≥ target (0.90), and ≥ 15 of the selected rows
  are minority rows (the lead's class is not the majority) with Wilson lower bound of agreement on those selected
  minority rows ≥ 0.80 (`--minority`; the fit report's `cv_minority_recall` is over all rows, not the selected ones).
- **One-sided** (noul only, ≥ 15 lead "yes"): largest-coverage τ (scan unique OOF p_yes descending, first that
  passes) with: ≥ 30 local (p_yes < τ), Wilson lower bound of "yes" recall among forwarded ≥ 0.85, Wilson upper
  bound of local miss rate ≤ 0.02 (OOF p_yes rounded to 4 decimals). Taken if its coverage beats the symmetric one.
- **One-sided yes** (`one_sided_yes`, noul only, ≥ 15 lead "yes" and ≥ 1 "no"): the mirror, for understudents that can
  only confidently confirm "yes" (e.g. LLM-written credential regexes that abstain otherwise). Answer locally only a
  confident "yes": local iff p_yes > τ (an abstention is never local). Largest coverage first (scan the unique OOF
  or held-out p_yes ascending, each value minus 1e-4 so that value's rows stay local; thresholds below 0.5 are not
  tried, so a local answer is the model's own argmax), the first τ that passes: ≥ 30 local, Wilson lower bound of "no"
  recall among the forwarded ≥ 0.85, Wilson upper bound of the local false-"yes" rate (local rows the lead
  answered "no") ≤ 0.02. Taken if its coverage beats the best of the other modes (symmetric, then one-sided "no").
  Applies to every understudent (shared graduation code). Demoted while the Wilson lower bound of the false-"yes" rate in
  the audit window (≥ 20 audits) is > 0.02; `audits.misses` counts false "yes" answers.
- Minimum examples before trying to graduate: 60 (`--min-examples` overrides). With refits every 100 examples this rarely matters: the first refit sees ~100.
- **Novelty**: cosine nearest neighbor among the 1,000 most recent examples' novelty vectors; novel if below the
  5th percentile of those examples' own NN similarities.
- **Audits/demotion**: window of the last 100 audits (≥ 20). Symmetric demoted while Wilson upper bound of
  agreement < target; one-sided (either direction) while Wilson lower bound of miss rate > 0.02 (a miss is a local
  "no" the lead calls "yes"; for `one_sided_yes`, a local "yes" the lead calls "no").
- Wilson interval with z = 1.96 (two-sided 95%: `stats::wilson`, `wilson_z`).
- **Graduation policy** (`--graduation pointwise|sequential`, replay, judge, train, serve; stored in the snapshot, an existing
  snapshot refuses another; default `pointwise`, byte-identical to everything above and in a snapshot). Every refit re-tests
  graduation, so a question gets many chances to pass by luck. `sequential` spends alpha across a question's attempts:
  attempt `k` (1-based) tests every Wilson bound of the graduation scan at the two-sided level
  `alpha_k = 0.05 * 6 / (pi^2 k^2)` (the sum over all attempts is at most 0.05), i.e. `z_k = -Phi^-1(alpha_k / 2 / m)`
  (`stats::sequential_z`; 2.16 at k = 1, 3.24 at 5, 3.61 at 10, 4.15 at 30 for one candidate). It applies to every bar of
  the scan: the agreement and minority bars of the symmetric mode, the "yes"/"no" recall lower bound and the local miss
  upper bound of both one-sided modes, the `rules` tier's graduation, and the certificate of `--abstain calibrated`
  (which replaces its own Bonferroni z). `m` is the number of candidates raced in the attempt (a ladder's tiers, auto's
  candidates: Bonferroni on top of the spending; 1 for a single understudent). The counter is per question
  (`Understudent::attempts`, `attempts` in the snapshot's question entry, omitted while 0): a refit that runs the graduation
  test (at least `--min-examples` rows and two classes) increments it, a ladder's cheap-tier refits included (so the
  tiers that did not refit in that event are charged too, conservatively), and it is **never reset**: a demoted
  question that re-graduates continues its count, as does a question that stops and starts graduating. The refit's
  report (and replay's `refits[]`) carries `graduation: {policy, attempt, z, candidates}`. Demotion's audit bounds
  stay at z = 1.96. Not covered: the threshold scan within one attempt (50 thresholds, a pre-existing multiplicity) and
  the dependence between refits' overlapping windows (the bound is a union bound, so it holds regardless). The price is
  coverage: z rises with k, so a question that is only just above the target graduates later or loses graduation at a
  later refit.
  Measured (docs/graduation.md has the tables): in a synthetic harness (target 0.97, true agreement 0.95 to 0.965, 30 refits)
  the probability of ever graduating is 0.0% under `sequential` in every cell, against up to 1.9% pointwise (8.2% with four
  candidates, 20.3% with the threshold scan); with true agreement 0.99 `sequential` graduates about six refits later and
  is graduated for 69% rather than 92% of the refits. On the committed replays it matched pointwise agreement (100% on
  everything answered) at lower coverage (15 cells: pointwise graduated in 12, sequential in 4 of those 12).
  **Recommendation: keep `pointwise` the default** and use `sequential` where a false graduation is costly and the stream is
  long; the default is not flipped.

## Deep design

See [docs/understudents/deep.md](docs/understudents/deep.md). Encoder: vendored eg2cpu `Model::embed_batch_tokens(texts, layers)`, default layers `[4, 8, 12]` (flag). Text =
the state's text rendering (capped at 20,000 characters, head 2/3 and tail 1/3), then **chunked** when it exceeds
3,000 characters (see "Long states"; `--no-chunk` clips it instead, head 2/3, tail 1/3
of 3,000 chars).

Readouts (all per question):
1. **pool**: for each layer, token mean and token max (per dimension) → concatenate (2 × 512 × |layers|),
   standardize (mean/sd of the training examples), L2-regularized weighted multinomial logistic regression
   (own implementation, e.g. full-batch L-BFGS or Adam; no Python).
2. **attn**: on one layer (default the middle of `layers`), K = 4 learned query vectors; for each, softmax
   attention over tokens gives a pooled vector; concatenate the K pooled vectors with the token mean → linear
   softmax head. Trained jointly with Adam on weighted cross-entropy + L2, a fixed number of epochs, seeded.
   Token states are stored as f16/bf16, capped at 512 tokens per example (keep head and tail).
3. **pair** (when question.json has `pair`): embed `state[a]` and `state[b]` separately; features become
   `[x, y, x − y, x ⊙ y]` of the pool features (and of the attn pooled vectors for attn).
4. **auto** (default): fit pool and attn with the same folds, pick by OOF agreement with the one-standard-error rule
   preferring `pool` (cheaper). Graduation uses the chosen readout's OOF predictions.

Novelty vector: L2-normalized token mean at the middle layer (pair: of the concatenated pair).

## Understudents

One page per design, with diagrams and examples, in [docs/understudents/](docs/understudents/README.md).

Several understudents run under the identical routing and statistics above (5-fold OOF, Wilson graduation, novelty
forwarding, audits, demotion, the 100-example refit schedule, `--min-examples`, seeds). `--understudent NAME` on `replay`,
`judge`, `train` and `serve` picks one (default `deep`); it is stored in the snapshot, and an existing snapshot refuses another. A
understudent supplies only features, a fit (final model plus OOF probabilities), a prediction and a novelty vector.
`readout` in a decision names what answered; the replay summary carries `understudent`, `coverage_by_readout` and every
refit's timings (`refits[].secs`, `fit_secs`; a ladder also `refits[].tiers.<tier>` and, in the summary,
`coverage_by_tier`, `graduated_at_item_by_tier`, `refit_secs_by_tier`).

| `--understudent` | features | model | encoder | novelty vector | `readout` |
|---|---|---|---|---|---|
| `deep` | the `deep` section: layers 4/8/12, mean+max pool or attention (`--readout`, `--layers`) | pool LR / attn head, auto | to layer 12 | L2-normalized layer-8 mean | `pool`, `attn` |
| `probe` | layer-8 token mean only | standardized LR, C=0.5 | to layer 8 | L2-normalized layer-8 mean | `mean` |
| `keywords` | hashed char 3-5-grams (lowercased) + word uni/bigrams of the full state text (capped at 20,000 chars) | sparse LR | **none** (the model is never loaded) | 256-d projection of the hashed vector | `keywords` |
| `lite` | the input token-embedding table's rows (`Model::input_embeddings`), token mean and max | standardized LR, C=0.5 | no layer runs | L2-normalized mean of the embeddings | `lite` |
| `neighbors` | the L2-normalized layer-8 mean | cosine 7-NN vote | to layer 8 | the same vector | `neighbors` |
| `ladder` | tiers (`--tiers`, default `keywords,lite,probe,deep`), each as its own row above | one fit per tier, each with its own graduation, threshold, novelty, audits and demotion | only up to the tier that answers | each tier's own | `keywords`, `lite`, `mean` (probe), `deep:pool`, `deep:attn` |
| `shape` | cheap structural features of the state (below): lengths, shares, markdown, URLs, refusal/apology/hedging markers, echoed numbers, JSON field types and values | standardized (clipped at ±6 sd) LR, C=0.5 | **none** (the model is never loaded) | the standardized feature vector, L2-normalized | `shape` |
| `blend` | one design over [`keywords`'s hashed features] ⊕ [`shape`'s features] ⊕ [`lite`'s embedding mean+max] | a single sparse+dense LR (`keywords`'s solver and C=30; blocks scaled to unit norm) | the embedding table only, no layer runs | `keywords`'s projected vector | `blend` |
| `memo` | the whole state text's hash and a 128-hash MinHash of its character 5-grams | retrieval over the window's lead answers (below); abstains unless an exact or near-duplicate match | **none** | none (it only answers near-duplicates) | `memo` |
| `net` | deep's pooled encoder features (layers 4/8/12, mean+max, chunked; pairs the 4-way features) ++ `shape`'s structural features, each block standardized on the fitting rows only (after the inner split) and clipped at ±6 sd | one hidden layer (64 ReLU, dropout 0.1), Adam, early stopping on an inner split of each training fold, seeded | as deep (to layer 12; shares its encode-cache entries) | L2-normalized layer-8 mean | `net` |
| `checklist` | sparse binary features: `keywords`'s hashed buckets present + `shape`'s indicators (refusal/apology/hedge markers, presence bits, JSON types, length buckets, ...) | an ordered decision list of one-literal rules, greedily induced; abstains when no rule fires | **none** | `keywords`'s projected vector | `checklist` |
| `auto` | races `--candidates` (default `keywords,lite,probe,deep`) at every refit and serves ONE | one fit per candidate on the shared window | only the serving candidate's | the serving candidate's | the serving candidate's (`keywords`, `lite`, `mean`, `deep:pool|attn`) |

- **deep** is the behavior described in "Deep design".
- **`probe`** is the plain linear probe: `linear.rs` minimizes sklearn's `LogisticRegression(C=0.5, lbfgs, L2)`
  objective (`C·Σ wᵢ lossᵢ + ½‖w‖²`, intercept unpenalized; solved scaled by `1/Σ wᵢ`). The encoder stops at layer 8
  and no token states are stored. `--layers`, `--attn-layer`, `--readout` apply only to `deep` and a ladder's deep tier.
- **keywords** features: for each state text (a pair's two texts are salted apart and summed), lowercase character
  3-, 4- and 5-grams and word (alphanumeric run) unigrams and bigrams are hashed (FNV-1a, splitmix finalizer) into
  2^18 buckets with a sign from another hash bit; counts per distinct feature get sublinear tf (`1 + ln c`), signed
  values accumulate per bucket, the vector is L2-normalized. The model is the same weighted L2 logistic regression
  as `probe` on the sparse vectors, with C=30 (L2-normalized inputs need far more than C=0.5) and the weights of
  buckets seen in fewer than 2 training rows fixed at zero; L-BFGS to max-gradient 1e-3, 300 iterations at most. The
  fit and its five folds run in parallel (a 2,000-example refit takes about 0.15 s on ~150-character texts, 0.9 s
  on 2,400-character ones). Novelty is a fixed sparse random projection of the hashed vector to 256 dims (each
  bucket adds ± its value to 4 hashed dims), L2-normalized, so the nearest-neighbor cosine of references and
  query approximates the sparse cosine at a fixed cost.
- **shape** (`src/shape.rs`) is a calibrated logistic regression over `DIM` = 430 structural features and no encoder. Per
  field slot (16 of them, **keyed by field name**: a stable FNV-1a hash of the name picks the slot, so adding or removing
  a field never shifts another field's slot, and a missing field is visible as its own slot's presence bit dropping; on
  a hash collision the field with the smaller name keeps the slot, whatever the order the fields came in; a non-object
  state is one field named ""; fields that lose a collision still count in the overall block and in the echo features):
  presence, the JSON type (null/bool/number/string/list/object),
  the numeric value (signed log, clipped at 1e9) and the boolean value, then 16 text features of the field's text: log
  chars, words and lines; the share of digits, uppercase letters and punctuation; log counts of markdown headers,
  bullets and code fences, of URLs and of question marks; log counts of refusal, apology and hedging markers (a fixed
  lexicon of substrings on the lowercased text, `REFUSAL`/`APOLOGY`/`HEDGE` in `shape.rs`); the log count of numbers; the
  share of the field's numbers that another field also holds (an answer echoing its question). The overall block has
  the same 16 features of the whole state text (its echo feature is the share of numbers held by at least two fields),
  the log number of fields, the state's own type, the log ratio of char lengths for each pair of the first four
  slots, and the log length ratio of the question's `pair` fields (0 without a pair). Features are standardized with the training rows' mean and population sd and clipped at ±6 sd (a column that
  was constant in training cannot blow up), then `linear.rs`'s logistic regression at its C (fixed 0.5, no inner CV:
  the 430 columns on <= 2,000 rows barely move with C). Novelty is the standardized, clipped vector, L2-normalized.
  With 16 slots two of a state's fields sometimes collide and only one keeps its per-field columns.
  **Out-of-fold fits standardize per fold:** each CV fold's standardizer (and clipping) is fitted on that fold's
  training rows only, so the out-of-fold predictions that decide graduation never see a held-out row's statistics; the
  final model's standardizer is fitted on all rows.
- **blend** is the "richer features" design in one model. Row = [`keywords`'s hashed, L2-normalized sparse vector] ++
  [`shape`'s features, standardized, clipped, then normalized to unit L2 norm per row] ++ [`lite`'s mean+max embedding
  pools (pairs: the 4-way pair features), standardized, clipped, then unit L2 norm per row], so each block has norm
  exactly 1 (0 for an all-zero block), as the n-gram block does, whatever outliers the clipping lets through; each CV
  fold's two standardizers are fitted on that fold's training rows only (the final model's on all rows);
  the dense blocks are extra columns after the 2^18 buckets of the same sparse solver (`keywords::fit_dim`), with
  `keywords`'s C=30, MIN_DF and L-BFGS settings. It stores the two standardizers (`Fit::blend`) and the model in
  `Fit::sparse`. Only the embedding table is read (`lite`'s encode path). Novelty is `keywords`'s projection.
- **memo** (`src/memo.rs`) is retrieval over the window's lead answers, with no model and no encoder. A state's key is
  the hash of its whole text and a MinHash signature (128 hashes) of its character 5-gram set; the similarity is 1.0 for
  an exact match, else the share of equal signature positions (an estimate of the Jaccard similarity, sd about 0.02
  near 0.95). Given a new state it takes the stored states at similarity >= `thr_sim`; if there are none, or they
  disagree on the label, it abstains (`proba` is None: the request is forwarded, never local); else it returns the lead
  answer of the most similar (the newest among equals). The stored answer's probabilities are rebuilt from the label
  and the example weight (the lead's confidence; 1.0 for a labeled example), which is exact for a two-class question.
  `thr_sim` is chosen by the fit's own CV from {0.80, 0.85, 0.90, 0.95, 1.0}: out-of-fold, each row is answered from
  the other folds' rows alone, the shared graduation test (Wilson bounds on the agreement over the rows it answers, an
  abstention being neither agreement nor disagreement) is run for each, and the value whose answers graduate with the
  most coverage wins (ties: the stricter); if none graduates the default 0.95 is kept (and nothing is answered). The
  out-of-fold answers that graduation certifies use **nested grouped CV**: for each outer fold the threshold is picked
  from inner folds within that fold's training rows alone, and the fold's rows are answered at that threshold, so no
  evaluated row's label influenced the threshold used on it (the deployed `thr_sim` is the same scan over all outer
  out-of-fold rows). The report carries `memo_sim`, `memo_sim_grid`, `memo_sim_nested` (the per-fold picks), `answered`; `cv_agreement` is over the answered rows (0.0 below 30 of
  them, so memo never props up an auto race's give-up check). Memo needs the shared bars to be met on its own
  answers, so with no duplicates it never graduates; with duplicates it answers exactly where reuse is reliable. A state
  forwarded to the lead joins the stored keys at the next refit (in a ladder, every 25 examples). No novelty
  floor: a near-duplicate is by definition not novel.
- **net** (`src/netdesign.rs`) is a small trained classifier: input = deep's pool features (`pool_dim`: mean and max over layers
  4/8/12, 3,072 dims for a one-field state) ++ `shape`'s 430 features; each block is standardized with the TRAINING rows'
  mean and sd and clipped at ±6 sd (per fold, like `shape` and `blend`: a held-out row never reaches its fold's
  preprocessing); one hidden layer of 64 ReLU units with dropout 0.1, a softmax over the classes seen in training,
  weights = the lead's confidence, L2 penalty 1e-4, Adam (lr 2e-3) on mini-batches of 32. **Early stopping:** 15% of the
  training rows (whole id groups, chosen by a hash of the group and the seed) are an inner split; the epoch with the
  lowest weighted validation loss wins (patience 5) and its weights are restored; with fewer than 20 fit rows, 10
  validation rows or a single class on either side everything trains for a fixed 15 epochs at most. **Deterministic and
  bounded:** a fixed seed per fit (`seed * 1000 + fold`: initial weights, batch order, dropout masks), single-threaded
  fits run in parallel (the final model and the five folds), and epochs are capped by a multiply-accumulate budget
  (1.6e10, at most 40 epochs, at least 6) rather than the clock, so a 2,000-row refit takes a few seconds and a replay
  repeats exactly. It reads the same encoding as deep, so `--layers`, `--no-chunk` and the encode cache apply and cache
  entries are shared with deep; `readout` is `net`. The report carries `net_epochs` (kept epoch per fit: final, then
  the folds) and `net_dim`.
- **checklist** (`src/checklist.rs`) is an interpretable decision list, no encoder. A state's binary features are every hashed
  n-gram bucket it has (ids `0..2^18`, `keywords`'s hashing) plus about 440 indicators derived from `shape`'s vector with
  fixed thresholds (no fitting, so nothing leaks): per field slot the presence bit, JSON type, a true boolean, a
  positive or negative number; for every field's text and the whole state's a length bucket (<= 10, 11-50, 51-200,
  201-1000, > 1000 characters), multiple lines, markdown headers and bullets, code fences, URLs, question marks, numbers,
  refusal, apology and hedging markers, an echoed number, mostly digits and mostly uppercase. **Induction** is greedy
  sequential covering (CN2/RIPPER-style) with one-literal rules `IF feature THEN class`: among the training rows no
  earlier rule took, pick the (feature, class) with the highest Wilson lower bound of precision (ties: more rows, then
  the lower feature id, then the lower class) among features in at least 5 rows, keep it when the bound is >= 0.6 and it
  takes >= 5 rows, drop the rows it covers, repeat up to 40 rules, so rules come ordered by precision on the rows left to
  them. The default class (the majority of the rows no rule covers) is reported but **never answers**: the understudent
  answers only when a rule fires (the first firing rule's class at confidence `(hits+1)/(rows+2)`, the rest of the mass
  spread over the other classes) and otherwise abstains (`proba` is None, the request is forwarded). Out-of-fold, each
  fold's list is induced on that fold's training rows and a row nothing fires on is an abstention (a NaN row, as
  `memo`); graduation is the shared Wilson scan over the rows it answers and `cv_agreement` is over those rows (0.0
  below 30). It is very fast (a 600-2,000-row refit took 0.3-0.7 s on a loaded 8-core box, no encoder) and is on the cheap refit schedule.
  **Printing rules:** the report's `checklist` has `rules` (feature, class, confidence, rows, hits), `default` and a
  `listing`, one rule per line, e.g. `1. IF [answer].refusal THEN yes   [conf 0.974, 41/42 rows]`. A hashed n-gram prints
  as its text (`"sorry"`, `word:refund`, `words:i cannot`) when the training states' texts are known (the encodings of a
  running proxy carry them in memory, `Encoding::aux`; they are never saved in a snapshot), the most frequent n-gram of
  the bucket, annotated `(+N colliding)` when several share it, else as `gram#<bucket>`; a structural slot prints as the
  field that most often holds it (`[answer].refusal`, else `slot7.refusal`); the whole-state block prints as `all.*`.
  `understudy status` prints each question's `checklist` (also in its JSON) and the top rules on stderr; the listing is
  saved in the snapshot's fit report, so it survives a reload.
- **lite** needs no transformer layer: tokenizer plus table lookup, microseconds per item.
- **neighbors** votes with the 7 most similar training rows (earlier row wins ties), each weighing `max(cos, 0)`;
  every class gets +0.5; the votes are normalized. Lead confidence weights are not used. OOF probabilities use
  each fold's training rows.
- **ladder** has one **tier** per entry of `--tiers` (any of `keywords`, `lite`, `probe`, `deep`, each at most once,
  cheapest first). Every tier trains on the same example window and is a complete understudent of its own: its own fit and
  5-fold OOF, graduation (Wilson bounds, `--target` and `--minority` apply to every tier), threshold and mode,
  novelty references, audit window and demotion. Per request and question:
  1. Tiers are tried in order. A tier is *eligible* when graduated, confident per its mode and not novel, and *may
     answer* when it is also not demoted. The **cheapest tier that may answer answers.** A tier's features are
     computed only when a request reaches it: `keywords` needs no model; `lite` only the input embeddings; `probe`
     layers 0..8; `deep` up to layer 12. Only tiers with a fit are tried, so a learning tier costs a prediction, not
     a skipped forward.
  2. `probe` and deep share one forward pass when their texts agree (states under 3,000 characters): `probe`'s pass runs
     the layers of deep's list up to 8 through eg2cpu's `embed_batch_tokens_partial` (the patch is in
     `vendor/eg2cpu/PATCHED.md`) and keeps the partial state in the request; if `probe` can't answer, `resume_batch_tokens`
     continues the same pass to layer 12 instead of recomputing. Longer states (`probe` clips, `deep` chunks) are encoded
     separately.
  3. If no tier may answer, the request is forwarded to the lead. The example needs every tier's features, so all
     are computed and every tier with a model predicts; each one that was eligible (including a demoted one) records
     an audit against the lead's answer. A local answer is audited at `--audit-rate`: the answering tier and any
     other tier whose prediction was already computed record audits (later tiers are not computed to be audited), and
     the item becomes an example with every tier's features.
  4. A demoted tier is skipped, so its answers go to the next tier; it keeps getting audits whenever it is reached
     and eligible, and `install` re-evaluates demotion on every refit, as for any understudent.
  5. **Refits:** a refit event fires when a question has 100 new examples (all tiers refit) and the cheap tiers
     (`rules`, `keywords`, `lite`, `shape`, `blend`, `memo`) also refit every 25 new examples
     (`Settings::cheap_refit_every`; ladders only: `auto` and single understudents refit every 100). This is what lets a
     cheap tier keep learning, and take over, after a stronger tier started answering and slowed the arrival of examples.
  A decision's `readout` names the answering tier (`keywords`, `lite`, `mean`, `deep:pool|attn`); a forwarded item
  reports the first eligible tier, else the last tier that predicted, in `understudent_label`/`mode`/`eligible`.
  `understudent_us` covers every feature and prediction computed to route the item (a forwarded item pays for all tiers).
  The replay summary reports `coverage_by_tier`, `graduated_at_item_by_tier` and per-tier refit seconds.
- **auto** ("race, then serve ONE") fixes the ladder's error compounding (each tier's mistakes add up, so a ladder
  can be worse than its best tier alone). Candidates (`--candidates`, same rules and extension point as `--tiers`)
  are fitted at every refit on the same window with the same folds, each with the standard graduation test at
  `--target`/`--minority` on its own OOF predictions. The **serving understudent** is, among the graduating candidates,
  the one with the highest `cv_coverage`; candidates within one standard error of that coverage proportion
  (`sqrt(c(1-c)/n)`, `n` = rows fitted) are tied and the one with the lower *cost class* wins, then the earlier in
  the list (keywords < lite < `probe` < deep). Cost is measured per item: the candidate's encode time (moving average
  over the batches that computed it; deep adds `probe`'s, because a shared pass is timed in two pieces) plus its
  prediction time on recent examples. Costs are compared by decade of microseconds (`floor(log10 us)`), so timing
  noise cannot flip a choice and a replay repeats; `Settings::cost_us` fixes the costs for tests.
  - **`--prefer-fast EPS`** (auto only; default off, and with it off nothing above changes, byte for byte, the snapshot
    included) replaces the tie rule: among the candidates that graduated (so each met the agreement and minority bars
    on its own out-of-fold predictions), those whose `cv_coverage` is within EPS (absolute, e.g. 0.05) of the best form
    the band, and the **cheapest by modelled cost** serves, then the earlier in the candidate list. The cost is NOT
    measured: it is a recorded cost model, `Settings::prefer_fast` stored in the snapshot, so a replay repeats exactly.
    `model_cost_us(design, deepest layer run, tokens)` = a fixed per-design cost (`keywords` 130, `lite` 190, `shape` 80,
    `checklist` 170, `blend` 350, `memo` 100, `rules` 50 us; `probe` 200, `deep` 400, `net` 600) plus, for an encoder
    design (`probe`, `deep`, `net`), 50 us x tokens x the deepest encoder layer it runs (`probe` reads layer 8; `deep` and
    `net` run to their deepest layer, 12 by default). `tokens` is the mean token count of the 32 newest examples'
    recorded encodings (a deterministic function of the text; about 4 bytes a token when none was recorded, 256 when
    there is no example). The constants are the benchmark machine's: they set the costs `race` records (`cost_us` per
    candidate holds the modelled cost while the flag is on), but a decision depends only on their order. Settings given
    in `Settings::cost_us` still override. The tie rule's standard error plays no part. Wall-clock time never enters the
    decision. Flags: `--prefer-fast EPS` (0 to 1), `--understudent auto` only; a stored snapshot refuses a different EPS.
  - Only the serving understudent answers; its audits, graduation mode and demotion apply. Switching serving understudent
    (including to none) or mode clears the audit windows involved, as a mode change does for any understudent. Each refit
    records the race (`race`: per candidate `cv_agreement`, `cv_coverage`, `mode`, `why_not`, `cost_us`) in the
    refit's replay record, the replay summary's `refits`, `understudy status` and the snapshot; the summary adds
    `candidates`, `serving`, `serving_history` (`{item, understudent}` each time the serving understudent changed; it serves
    from that item on) and `lead_only`. `readout` is the serving understudent's.
  - **Lazy features, and their cost.** A local prediction computes only the serving understudent's features (a serving
    `keywords` never runs the encoder). A forwarded item (the lead is called anyway) and an audited local item
    compute every candidate's features, since each becomes a training example for all of them. So an `auto`
    question pays for all candidates on every example it learns from (the learning phase, audits, forwards) but only
    the serving understudent's price on local answers; while nothing graduates it costs a ladder's forwarded-item price.
  - **Lead-only verdict.** After `--give-up-after` examples (default 600) with no candidate graduated and none
    within 0.05 of `--target` in `cv_agreement`, the question is `lead_only` (status shows the reason): it
    forwards every request without computing features (the states are kept and encoded at the re-check), and
    refits only after every 1,000 new examples (one refit of every candidate, which can lift the verdict).
  - Examples for which some candidate lacks features (made while lead-only, or loaded from an old snapshot) sit
    out a refit until they are encoded.

### The `shape`, `blend` and `memo` understudents as ladder tiers and `auto` candidates

(`net` and `checklist` join them the same way in wave 2: `--candidates keywords,lite,probe,deep,shape,blend,memo,net,checklist` or any subset; the
defaults are unchanged. `net` is on the long refit schedule and uses deep's featurizer, slot and encode-cache entries, so a
race holding both encodes once; `checklist` is cheap and, like `keywords`, computes no model features. Both are opt-in.)

`--tiers` and `--candidates` accept `shape`, `blend` and `memo` anywhere (`--candidates keywords,lite,probe,deep,shape,blend,memo`);
the default tiers and candidates (`UnderstudentKind::default_tiers`) do not include them. Candidates tie
by cost class and then list order, so the cheaper ones should come first. In a ladder all three are on the cheap refit schedule
(every 25 examples), since none runs a transformer layer. An encoding gains `shape` (the vector) and `memo` (the key);
their encoding keys also hash the state JSON, because the features read the state's structure and not only its text
`blend` shares the embedding slot with `lite`, and its
sparse vector with `keywords`, so a race computes each input once.

`auto` over the extended candidates behaves like the default race whenever deep or `probe` wins, so the default candidates stay as they are and the three are opt-in.

### Abstention policies (`--abstain`, `src/abstain.rs`)

`--abstain default|calibrated` (replay, judge, train, serve; stored in the snapshot, an existing snapshot refuses another)
picks how out-of-fold predictions become a graduation threshold, for ANY understudent, ladder tier or auto candidate (the
`rules` understudent, which validates on held-out rows rather than out-of-fold, keeps its own scan).

- `default` is the symmetric scan above: one threshold, the lowest `t` in 0.50..0.99 that meets the bars.
  Known limitation of `default` (unchanged, deliberately): it tallies the confidence-selected out-of-fold rows, but
  serving also forwards novel rows (`Fit::novel`), and the final model is refitted on all rows, so the answered population
  it certifies is not exactly the one served (removing novel rows need not preserve agreement). `calibrated` has no such
  gap.
- `calibrated` is a **certified frozen policy**: the certificate measures exactly what is served.
  1. **Group-disjoint three-way split** of the labeled rows, TRAIN / SELECT / CERTIFY, about 60/20/20 by group (a `None`
     group is its own; seeded by `--seed`; `abstain::three_way`). With fewer than 3 groups no split exists and the policy
     never graduates (`why_not`).
  2. The model is fitted on **TRAIN alone**: every preprocessing step, the early stopping (net) and the novelty reference
     set and floor. There is no cross-validation in this mode (and so no out-of-fold predictions): `deep` fits its pool
     readout unless `--readout attn` forces attention, and `memo` keeps its default similarity.
  3. The per-class thresholds (a symmetric answer is local when the confidence of the predicted class `c` is at least
     `thr[c]`, `Fit::class_thr`) are chosen on **SELECT**, with the novelty gate already applied, maximizing the answered
     share subject to the usual bars (at least 30 answered, the Wilson lower bound of agreement >= `--target`, at least
     15 answered minority rows with the Wilson lower bound of their agreement >= `--minority`; "minority" is relative to
     TRAIN's majority). Two supported classes are searched exhaustively (50 x 50 grid, ties to the stricter), more by
     coordinate ascent. **A class needs `MIN_CLASS_SUPPORT` = 10 answerable SELECT rows predicted as it; otherwise its
     threshold is infinite (a null in the snapshot) and it is never answered.** There is no selection margin and no nested
     threshold search.
  4. The **exact frozen serving predicate** (model, thresholds, novelty floor: `Serving::predict`'s `eligible`) is
     evaluated **once on CERTIFY**. The policy graduates only if the Wilson bounds on CERTIFY's answered rows meet the same
     bars, with the same minimum counts. Otherwise the fit does not graduate, and it never falls back to the default
     scans, the one-sided modes or any other uncertified policy.
  5. Serving uses that frozen model and those thresholds: nothing is refitted after certification. Every refit is a new
     model, split afresh and certified afresh. `cv_agreement` and `cv_coverage` in the report are CERTIFY's agreement and
     answered share (so auto's race compares certified coverage); `split`, `select` and `certificate` (`rows`, `answered`,
     `agreed`, `agreement`, `wilson_low`, `z`, `minority_answered`, `minority_agreed`, `coverage`, `passed`) carry the evidence.
  6. **Several candidates, one CERTIFY set.** A ladder's tiers and auto's candidates are certified on the same rows, so
     each must pass at a Bonferroni-corrected z (`abstain::bonferroni_z(m)`, the two-sided level 0.05/m: 1.96 for one
     candidate, 2.24 for two, 2.50 for four), and the choice among them is no selection on CERTIFY noise.
  - **Costs and caveats.** The served model sees only the TRAIN share (60%) of the labeled rows, so it is weaker than the
     default's all-rows refit, and the CERTIFY share (20%) makes the evidence thin: with 20% of the labeled rows the 30 /
     15 answered counts need roughly 150+ rows. The certificate is per attempt: a question refitted again and again gets
     repeated chances to pass by luck, which the audit window and demotion (unchanged) remain the backstop for, and which
     `--graduation sequential` bounds by spending alpha across the attempts (replacing the Bonferroni z above by the attempt's);
     per-class reliability is not certified, only the aggregate. net's standardizers are now fitted after its inner
     early-stopping split, on the fitting rows only (this also changes the default-policy `net`, a bug fix; no other
     understudent's default changes).

### Wave-2 validation (`net`, `checklist`, `--abstain calibrated`)

Local replays only (recorded lead answers, no API calls): `understudy replay --understudent U [--abstain calibrated]` on the
streams `understudy-bench` builds (items in order with the lead's recorded probabilities), the shared encode cache,
defaults otherwise (target 0.90, minority 0.80, audit rate 0.10, seed 0). `data/{passage_answers,tool_failed,incident_severity}`
with lead Jev. Each cell is coverage (share
of items answered locally) / REALIZED agreement with the lead's recorded answer on those (Wilson lower bound in brackets), measured
over the whole online replay, graduation included. `auto` is the default race (`keywords,lite,probe,deep`). This is the
run of the certified calibrated policy above (the earlier nested-CV policy's tables are superseded and removed).

#### incident_severity (lead jev)

| understudent | default: coverage / agreement (Wilson low) | calibrated: coverage / agreement (Wilson low) |
|---|---|---|
| deep | 11.3% / 91.2% (82.1) | 0.0% / n/a |
| keywords | 3.2% / 100.0% (83.2) | 0.0% / n/a |
| blend | 0.0% / n/a | 0.0% / n/a |
| net | 0.0% / n/a | 0.0% / n/a |
| checklist | 0.0% / n/a | 0.0% / n/a |
| auto | 11.3% / 91.2% (82.1) | 0.0% / n/a |

#### passage_answers (lead jev)

| understudent | default: coverage / agreement (Wilson low) | calibrated: coverage / agreement (Wilson low) |
|---|---|---|
| deep | 14.5% / 97.7% (92.0) | 0.0% / n/a |
| keywords | 0.0% / n/a | 0.0% / n/a |
| blend | 0.0% / n/a | 0.0% / n/a |
| net | 0.0% / n/a | 0.0% / n/a |
| checklist | 0.0% / n/a | 0.0% / n/a |
| auto | 14.5% / 97.7% (92.0) | 0.0% / n/a |

#### tool_failed (lead jev)

| understudent | default: coverage / agreement (Wilson low) | calibrated: coverage / agreement (Wilson low) |
|---|---|---|
| deep | 25.3% / 98.0% (94.4) | 0.0% / n/a |
| keywords | 11.8% / 98.6% (92.4) | 0.0% / n/a |
| blend | 0.0% / n/a | 0.0% / n/a |
| net | 27.3% / 95.1% (90.7) | 0.0% / n/a |
| checklist | 0.0% / n/a | 0.0% / n/a |
| auto | 25.3% / 98.0% (94.4) | 0.0% / n/a |

Reading it:

- **The certified policy is far more conservative, as it should be.** It fits on 60% of the labeled rows and graduates only
  when 20% of them certify the frozen policy, so it graduates later and covers less than the default: on tool_failed,
  passage_answers and incident_severity no calibrated configuration graduated at all (the default graduates `deep` on all
  three). Its earlier tool_failed gains (the single threshold's 11.8% -> 31.2% for `keywords`) came with uncertified
  thresholds and are gone.
- **A calibrated configuration can still realize a Wilson low under the 0.90 bar**: the certificate passes on a thin CERTIFY
  set, and then too few items are served for the audit window to act. This is the per-attempt luck the caveat above names.
  The certificate bounds a single attempt on CERTIFY's rows, not the realized agreement of every refit that follows.
- **`net` and `checklist` win no question.** Default `net` is 0.0% on incident_severity and passage_answers and 27.3% / 95.1%
  (90.7) on tool_failed, where `deep` is at 25.3% / 98.0%: it graduates there at a lower agreement. `checklist` (no encoder)
  gives rules a person can read but covers nothing on these three sets. The opt-in `auto-wide` race was not re-run.
- **The default policy's decisions are unchanged** by the calibrated policy's introduction (`--abstain default` is the
  untouched path, pinned by a test).

### Grouped CV with few groups

Ids carrying a group prefix (`<item>#<k>`) keep an item's siblings on one side of every CV split. With at least 5
groups (a `None`-group row counts as its own) the folds are 5-way `GroupKFold`. With 2-4 groups the CV uses
one fold per group (`k` = the number of groups); with fewer than 2 the understudent declines to graduate and records
`why_not: "fewer than 2 id groups: no grouped cross-validation is possible"`. It never falls back to an ungrouped
split when groups exist. Datasets without group prefixes use plain 5-fold CV.

### The `rules` understudent (`src/rules.rs`)

`--understudent rules`, or `rules` as the **first** entry of `--tiers` (`--tiers rules,keywords,lite,probe,deep`; the
default tiers do not include it). Decision rules an LLM writes as regexes, checked statistically against the lead:
the cheapest tier (microseconds; no model, no encoder; a ladder answering from rules computes no other tier).

- **Rule set:** ordered `{"pattern","field"?,"label","why"}` plus a `default` (a label or `"abstain"`). First match
  wins; `field` restricts the regex to one top-level state field. No match: the default answers, or the set abstains
  (the item is forwarded, never local). `regex` crate: linear time, no look-around.
- **Proposal** at a refit with >= 60 examples when there is no rule set, or the current one is not graduated and
  >= 100 new examples arrived since the last proposal; at most 5 proposals per question. The LLM (a small open-weights chat model,
  `--rules-llm-url` default the `[rules]` table of the lead config file, else Fireworks' public chat-completions endpoint with the key in `$FIREWORKS_API_KEY` (`RulesSettings::default()` in `src/rules.rs`; `--rules-llm-model` and `--rules-llm-key-env` likewise),
  default model in `RulesSettings::default()`, `reasoning_effort: low`, JSON out) sees the question
  spec plus <= 40 stratified labeled examples (texts clipped at 6,000 chars) from the OLDER half of the window.
  Re-proposals also show examples the previous set got wrong.
- **Rejection:** a pattern that doesn't compile (or exceeds the size limit / 400 chars), has an unknown label, or
  matches >= 95% of all window examples (too broad to be a rule) is dropped.
- **Validation** is on the examples the LLM did NOT see. Each rule's confidence is the Wilson lower bound of its
  precision on the held-out rows it answers (rows taken by an earlier rule excluded); rules below 0.5 (the lowest
  threshold graduation scans) are dropped and the rest re-evaluated; a default below it becomes "abstain".
  Graduation is the usual symmetric / one-sided Wilson scan over held-out rows where the set answers (abstentions
  are not local), using the answering rule's confidence as the prediction confidence. Coverage is the share of
  held-out rows answered locally. Audits and demotion work as for other tiers.
- **Determinism:** every LLM request/response is cached by sha256(request body) in `<snapshot>/rules_llm.jsonl` and,
  with `--rules-cache FILE` (JSONL), in FILE (loaded first, live calls appended). A replay with a warm cache makes
  no network calls and decides identically.
- **Reporting:** the active rule set with held-out validation stats (`fit.rules`: rules, dropped rules and why,
  heldout n/agreement/coverage, proposal count, request key, human-readable `listing`) appears in the snapshot, in
  `understudy status` and in the replay summary.

### Rules as an `auto` candidate

`--candidates rules,keywords,lite,probe,deep` (rules first, as for ladder tiers; the default candidates do not include rules). The rules candidate is fitted by its own path (proposal, held-out validation) while the others
use 5-fold OOF, but the race only uses two numbers per candidate, and both are out-of-sample estimates under the same
bars (target, minority, the symmetric and one-sided scans): `cv_coverage`, the share of evaluation rows answered
locally, and the graduation itself. For rules the evaluation rows are the held-out ones (the window rows the LLM never
saw), `heldout_n` of them, so rules' coverage is measured on a smaller set than the others'. The race therefore ties
candidates within one standard error computed over the smallest evaluation set among the graduated candidates (`n`
for the fitted ones, `heldout_n` for a graduated rules), which widens the tie band when rules competes; ties go to
the cheaper cost class, rules first. Rules examples need no encoder, so a race of `rules,keywords` computes no model features.
The rules LLM options, `--rules-cache`, the snapshot's rule bookkeeping and the summary's `rules_llm` / `rules` fields
apply as for any understudent, next to auto's `candidates` / `serving_history` / `serving` / `lead_only`.

### Long states

- `keywords` and `lite` read the **full** state text, capped at 20,000 characters (head 2/3, tail 1/3). `lite` does
  so in chunks (eg2cpu's tokenizer truncates at 1,024 tokens), always, regardless of `--no-chunk`.
- `deep` (and a ladder's deep tier) **chunks** by default; `probe` and `neighbors` keep clipping (the `probe` design
  is the plain-linear-probe control). `--no-chunk` makes deep clip as well. A text over 3,000 characters is split into consecutive
  chunks of at most 3,000 characters, each cut after the last line break in the final 30% of its window (a hard cut
  otherwise), at most 8 chunks (more would need a hard cut at 3,000 and a clip to 8 chunks; a 20,000-character cap
  gives at most 7). Each chunk is encoded on its own, then per field:
  - token mean = the token-count-weighted mean over chunks (one sum over all tokens divided by the total);
  - token max = the max over chunks;
  - attention tokens (layer `attn_layer`): the chunks' states in order, at most 512 in all. Each chunk keeps its fair
    share of the 512: chunks shorter than their share keep every token, the others split the rest evenly, and a chunk
    keeps the first two thirds of its share from its head and the last third from its tail;
  - the novelty vector is the L2-normalized combined mean at the middle layer.
  A short text is one chunk and encodes the same as with `--no-chunk`.

### Pair features for the cheap understudents

With `pair` in question.json: `lite` and `probe` pool each field separately and use `[x, y, x − y, x ⊙ y]` of the
dense features (as deep). `keywords` hashes each field into its own namespace (salted by field position), and adds the
signed difference of the two fields' features, hashed without the position salt into a third namespace (a feature both
fields share cancels; swapping the fields negates it exactly); the whole vector is L2-normalized.

## understudy CLI (Rust, clap)

Production has one mode: understudy sits in front of the lead and decides per request whether to answer or delegate;
it learns from every forward and keeps auditing after graduation. Its state is a **snapshot** directory.

- `understudy judge --snapshot DIR [--understudent deep|probe|keywords|lite|neighbors|rules|shape|blend|memo|net|checklist|ladder|auto] [--candidates ...] [--prefer-fast EPS] [--abstain default|calibrated] [--graduation pointwise|sequential] [--tiers keywords,lite,probe,deep] [--no-chunk] [--minority 0.8] [--question q.json] (--lead NAME [--lead-url URL] [--lead-model M] [--lead-kind systemone|chat] [--lead-key-env VAR] |
  --replay-lead stream.jsonl) [--audit-rate 0.1] [--save-every 100] [--layers 4,8,12] [--attn-layer L]
  [--readout auto|pool|attn] [--target 0.9] [--seed 0] [--min-examples 60]`: **the production command.** Reads
  `{"id","state"}` lines (or systemone requests `{"id"?, "questions": {name: question}, "state"}`; new questions
  start learning) on stdin and writes one line per input on stdout: a decision in the decisions.jsonl format (with
  `id`), or for several questions `{"id","answered_by","answers": {name: decision}}`; a failed line gets
  `{"id","error"}`. Routing as in replay; refits run synchronously every 100 new examples. Saves the snapshot every
  `--save-every` items and at end of input; prints a JSON summary to stderr.
  - `--frozen`: a fixed held-out test. The stream is scored with the snapshot's fits as they are and **nothing is learned
    from it**: no example is recorded, no audit counted, no refit run, and the snapshot is not saved (forwarded items still
    ask the lead, so the decision carries its answer). Default off.
  - Cold start: if DIR holds no snapshot, `--question` is required and a new snapshot is created (everything is forwarded
    until the question graduates). If DIR holds one, `--question` is optional and must match it.
  - **Leads** (`src/lead.rs`, `src/config.rs`, `src/chat.rs`). `--lead NAME` reads the preset `[leads.NAME]` from the lead
    config file; `--lead-url`, `--lead-model` and `--lead-kind` override it or stand in for it. No endpoint is built in. A preset
    has `kind` (`systemone`, the default, or `chat`), `url`, `model`, `key_env`, and for serve `route`; a chat preset also
    `reasoning_effort`, `json_schema`, `logprobs`, `confidence` and `max_tokens`. The file is `$UNDERSTUDY_LEADS`, else
    `$XDG_CONFIG_HOME/understudy/leads.toml`, else `~/.config/understudy/leads.toml`
    (`examples/leads.example.toml` documents every field). An unknown preset, or a missing config file, is an error that names where
    presets come from. Jev and Rune are example names (the System One leads the benchmarks used), not built-ins.
    A `chat` lead is an OpenAI-compatible `POST /chat/completions` endpoint driven through the adapter described under
    "Chat leads" below. `--lead-key-env VAR` (or the preset's `key_env`) names an environment variable whose value is sent as
    `Authorization: Bearer`; an unset variable is an error. The key is read from the environment only (never argv or a file),
    is held in memory, and is never printed, logged, put in a snapshot (a snapshot records `{"kind","url","model"}`) or in a capture log.
    Without lead flags, an existing snapshot's recorded lead (`http` or `chat`, by URL and model) is used, with its key from
    `--lead-key-env` if given. `--replay-lead` answers by id from a stream's `lead_probs` (offline, deterministic) and
    takes no lead flags.
  - Feature/fit flags (`--layers`, `--readout`, ...) only apply when creating a snapshot; on an existing one they must
    match it. `--audit-rate` may change.
- `understudy train --snapshot DIR [--question q.json] --examples ex.jsonl [--max-examples N] [lead flags as judge]`: **optional seeding**
  (backfill lead answers already paid for, or ground-truth corrections), on a new snapshot or one judge has used
  (load, add, refit once, save). Lines: `{"id","state","lead_probs"}` (a sampled example, as a forward),
  `{"id","state","truth": k}` (a labeled example: weight 2.0, kept apart from the sampled window, newest 5,000 per
  class, and sampled answers for labeled states are left out of fits), both, or
  `{"id","state"}` (the lead is asked; error without one). Prints a JSON report (n, CV agreement, minority recall,
  mode, threshold, CV coverage, readout, why_not) to stdout.
- `understudy model fetch [--revision REV] [--dir DIR]`: downloads the three model files from
  `<host>/google/embeddinggemma-2/resolve/<REV>/<file>` (host `https://huggingface.co`, or `$HF_ENDPOINT`; an optional
  `$HF_TOKEN` is sent as a bearer token to that origin only, and none is needed). Redirects are resolved as URL
  references against the current URL (relative, root-relative, scheme-relative and absolute `Location`s all work), and
  the token is dropped on any other origin. Error messages never include a URL's userinfo, query or fragment. Each
  file streams to a new, exclusively created temp file (`O_EXCL`, so a planted symlink is never written through) in its
  destination directory, with progress on stderr, and is renamed into place only if its size and SHA-256 match the
  pinned table; on a mismatch the temp file is deleted and the command fails. Files already present with the right
  hash are skipped. Without `--dir` it writes the hub-cache layout (`models--google--embeddinggemma-2/snapshots/<REV>/`,
  and `refs/main` for the pinned revision); with `--dir` it writes the three files into DIR (set `EG2_MODEL_DIR=DIR`
  to use them). `REV` is a full 40-hex commit; the default is the pinned revision. Any other revision has no built-in
  hashes: its LFS files (`tokenizer.json`, `model.safetensors`) must carry a valid hub `x-linked-etag` SHA-256 and are
  checked against it (a host that gives none makes the fetch fail, nothing is installed unverified), and `config.json`
  (a plain git blob) is reported unverified. Redirects are resolved as URL references (relative, scheme-relative and
  absolute `Location`s); the bearer token is sent only to the first origin, never to a redirect target on another one.
  Temp files are created exclusively (never through a pre-planted symlink), and error messages print URLs without
  their query, fragment or credentials. Prints the model directory on stdout.
- `understudy model status`: prints the resolved model directory, its revision and pinned revision, and `ok`,
  `MISSING` or `MISMATCH` for each file. It hashes ~1.5 GB (a few seconds); exits 1 unless all three match. It
  follows the same lookup as loading, including `EG2_MODEL_DIR` (revision unknown).
- `understudy status --snapshot DIR`: JSON per question (state `learning|graduated|demoted`, examples, mode, threshold,
  readout, CV agreement/coverage, audit window summary) plus lead and encoder fingerprint. Doesn't load the model.
- `understudy replay --question q.json --stream stream.jsonl --out decisions.jsonl [--understudent NAME] [--layers 4,8,12]
  [--readout auto|pool|attn] [--tiers ...] [--no-chunk] [--audit-rate 0.1] [--target 0.9] [--minority 0.8] [--seed 0]`: benchmarking; the replay above on a
  fresh in-memory snapshot; prints a JSON summary (understudent, graduation item, coverage, coverage by readout/tier, mode, readout, refits, timings)
  to stderr.

### `understudy serve`: the drop-in HTTP proxy (`src/serve.rs`)

`understudy serve [--bind 127.0.0.1:8787] [--mode forward|shadow|serve|race] [--snapshot-dir DIR | --no-snapshots]
[--capture FILE | --no-capture] [--capture-bodies] [--lead-url URL] [--no-model-download] [route flags] [understudent flags as judge]`
is a `POST /v1/systemone` server (alias `POST /systemone`; also `GET /healthz` and `GET /status`) that a System One caller
points at with only a base-URL change. Its decisions are non-streaming JSON; any other path gets a 404 ("Understudy handles only /v1/systemone; this path is not forwarded") and nothing is forwarded.

**Defaults.** `--understudent` defaults to `auto` here (`deep` for the other commands). `--snapshot-dir` defaults to
`understudy-snapshots` in the working directory, and a restart reloads it in any `--mode`; `--no-snapshots` keeps learning in
memory. `auto` never fails at startup for a missing encoder: it starts with the model-free designs (keywords, shape, checklist,
memo, and rules when a rules LLM is configured), loads EmbeddingGemma 2 if it is on disk or otherwise fetches it in the
background (`modelfetch`, checksum-verified; one log line when it starts and one when it finishes; `--no-model-download` skips
the fetch; a failure is a warning), and then adds lite, probe and deep to every namespace's race at its next job (`Proxy::
add_encoder_tiers`: the new candidates start unfitted and the existing fits, examples and serving candidate are kept). A snapshot
that was saved with the encoder needs it to load: the namespace waits for a fetch in progress and otherwise reports an error that
names `understudy model fetch`. `--candidates` still names the race (those that need the encoder join late).

**Upstreams.** A request is routed by its `model`. The routes are the leads of the lead config file (each with a `route`, else
its `model`, as the pattern: exact, or a prefix ending in `*`), plus `--route MODEL=URL` (System One) and
`--chat-route MODEL=URL` (chat), both repeatable and tried first; `--no-default-routes` ignores the file. `--lead-url URL` adds a
last catch-all route (named `lead`, pattern `*`) for any model, so no config file is needed. With no config, no `--route` and no
`--lead-url`, serve refuses to start and says so: there is no built-in default. `--route-key-env ROUTE=ENV_VAR` (ROUTE: a lead's
name or a route's model pattern) sends `Authorization: Bearer $ENV_VAR` for that route, as does a lead's `key_env` (a warning,
not an error, when that variable is unset). The key is read from the environment only. Without a configured key, a client's own
`Authorization` is forwarded. A **chat route** (a `kind = "chat"` lead, or `--chat-route`) keeps the caller's contract: the caller
sends System One, and serve answers it by chat calls (`chat.rs`), for forwards, audits and races alike. The route's `model`
replaces the request's in the chat call when set; a chat endpoint's error status and body are relayed; the response body is the
adapter's System One answer (not the endpoint's bytes). A route's chat settings are part of its namespace identity. An unknown model or a
body that is not a JSON object with a string `model` gets a 400 (`unknown_model` / `invalid_request`). A request body with
`Content-Encoding: zstd` (the TypeSafe SDK compresses bodies of 1 KiB or more) is decoded to be read; the upstream still gets
the original bytes and header. Another coding is a 400.

**Forwarding is transparent.** The original request bytes go upstream once (no internal retries, no redirects: callers
retry and have their own deadlines). The upstream's status, body bytes, `Content-Type`, `Retry-After` and other
headers (e.g. `fireworks-server-processing-time`) come back unchanged, whatever the status: 400 `max_tokens_exceeded`,
402 billing, 413/422, 429/503 with either `Retry-After` form, other 5xx and malformed JSON are all passed through, never
turned into 200s. Only a transport failure has no upstream response: it becomes 502 (504 on a timeout, `--timeout-secs`,
default 120) with `{"error":{"type":"upstream_unreachable",...}}`, which callers treat as retryable. Every response
carries `x-understudy: lead` (forwarded), `local` (answered here) or `error` (the proxy's own).

**Modes.**
- `forward`: pass-through plus capture; nothing is learned.
- `shadow` (default): forward every request and return the lead's response unchanged. Every successful answer is queued for
  the learners and the capture log records what the understudent would have answered, its eligibility and whether it agreed.
- `serve`: answer locally only when **every** question in the request is a known learner that is graduated and eligible
  (the routing of `judge`: tiers, confidence, novelty, demotion). Otherwise the whole request is forwarded as in shadow
  (mixed local/lead answers are not produced). A busy namespace (a refit holds it for longer than `--local-wait-ms`,
  default 5) forwards too. The local response is `{"model": <the request's model>, "answers": {<each submitted name>:
  {"type","choice","confidence","probabilities"}}, "usage": {"input_tokens":0,"cached_input_tokens":0,"output_tokens":0}}`
  (`noul`: `{"type":"noul","noul":p}`; `score`: `{"type":"score","score","confidence","probabilities"}`), HTTP 200 with
  `x-understudy: local`. A share `audit_rate` (`--audit-rate`, default 0.1) of local answers is also sent to the lead in the
  background (`--max-audits` at once) and learned from and audited against, so demotion still works; those calls cost the
  lead's price, nothing else does.
- `race` (latency first): every request goes to the lead **immediately** (the same transparent forwarding, on its own
  thread) while the local answer is computed on another thread, if every question is a known learner. The request thread
  returns the **first usable result**:
  - the local answer, when every question is graduated, confident, not novel and not demoted (the routing of `serve`) and it
    is ready before the lead's response: `x-understudy: local`, the same body as `serve`'s. The lead call is not cancelled:
    it finishes in the background and its response is never returned. It is queued as an audit (the served prediction against
    the lead's answer, at the fit that served it, as `serve`'s audits are) and a learning example, so every local answer in
    `race` is audited. A lead error (transport failure or a non-2xx status) after a local answer went out is only logged
    (`kind: "audit"` line with `status` or `error`; nothing is learned from it);
  - otherwise the lead's response, exactly as `forward` and `shadow` return it (bytes, status, headers, `x-understudy: lead`;
    errors pass through unchanged, a transport failure is 502/504). A forwarded request adds only what shadow adds (about
    0.3 ms): the lead call starts before any feature is computed, and the lead's response is returned the moment it arrives
    even if the local computation is still running (the abandoned computation's result is discarded). The understudent's
    would-be answer and its agreement with the lead are logged and learned from as in `shadow`.
  A local result and a lead result that are both ready are decided by whichever the request thread sees first; a lead
  error that arrives before the local answer is returned as the lead's error (the local answer is not waited for).
  `--race-max-inflight N` (default 64) caps the lead calls `race` has in flight (from the moment a request starts one until
  its response has arrived, including the ones that became background audits). A request that finds the cap reached behaves
  as in `serve` for that request: local when eligible (with `--audit-rate`'s audit, if any), otherwise forwarded after the
  local computation; it is counted in `race_capped`. `/status` adds `race_local`, `race_lead`, `race_capped` and
  `race_inflight`. Races on questions with no namespace yet (nothing learned) cost one thread for the lead call and one that
  returns at once.

**Learning is off the hot path.** The server is a thread per connection (`--max-conns`, default 512; more get a 503). A
request thread parses the body (to route and read the questions), forwards, answers the client, and only then hands the
rest to helper threads. Learning, refits and snapshot saves run on one worker per **upstream + model** namespace, each owning
its own `Proxy` (its own mutex, its own snapshot under `--snapshot-dir/<upstream>__<model>-<hash>`, saved every `--save-every`
learned requests, at shutdown and when idle). Rune and Jev, or `jev-latest` and `jev-preview`, never share a learner. The
queue is bounded (`--queue`, default 1024 per namespace); when it is full the job is dropped and counted (`learn_dropped`
in `/status` and the capture line), never waited for. A refit therefore stalls neither responses nor other namespaces.
Shutdown first waits (up to 30 s) for outstanding race lead calls and audits, so a late lead answer is still recorded as an
audit and a training example before snapshots are saved; anything still running after that is logged as abandoned.

**Namespaces and limits.** A namespace is route name + model + normalized upstream URL + a hash of the route's configuration
(never its key), recorded in the saved snapshot. A snapshot directory is named by route and model only, so a snapshot whose recorded
upstream or configuration differs is refused when the namespace loads (a clear error in `/status`, the log and the capture
line; the requests are forwarded and nothing is reused). A namespace exists only after a 2xx upstream answer with an `answers`
object for that model; `--max-namespaces` (default 16) and `--max-learners` per namespace (default 256) cap them, and beyond
a cap a request is forward-only and counted (`namespaces_refused`, `learners_refused`). A request has `--read-deadline-secs`
(default 60) in total to send its headers and body and a response `--write-timeout-secs` (default 30) to be taken; more than
32 trailers is refused. Framing is strict: duplicate or conflicting `Content-Length`, `Transfer-Encoding` with
`Content-Length`, any coding but `chunked`, bare line feeds and a chunk not followed by CRLF are errors that close the
connection. The upstream response bytes are relayed raw (no decompression, `Content-Encoding` kept, every value of a repeated
header kept, `Connection`-nominated headers dropped); one over `--max-response-mib` (default 256) is a 502 with a capture note,
never truncated. The caller's end-to-end request headers (`Idempotency-Key`, `Accept`, correlation ids, ...) go upstream; only
hop-by-hop ones, `Host` and `Content-Length` do not. 204 carries no `Content-Length`; 304 keeps the upstream's. Local
probabilities are apportioned to sum to exactly 1 at four decimals. A background audit records the prediction that was
served and the fit version that made it and agrees against those (`served` in the capture line; an audit across a refit is
logged but not counted toward demotion). Post-response work that finds the helpers' queue full is dropped and counted
(`post_dropped`), never run on the connection thread.

**Identity.** A learner is `namespace + spec hash` (`Question::key`: type, instructions, criteria; criteria order does not
matter, and each learner keeps its first option order). The request's question names are kept apart from it: `decision`
(Rust) and `action` (Go) are only names, identical specs under different names (even in one request) share one learner,
and `answers` keys always equal the submitted names. A question the learners don't parse (an unknown `type`) is opaque:
forwarded as is, never learned, and any request containing one is forwarded whole. A lead answer that is a bare
`{"choice": c}` (no probabilities) teaches as the label `c`; one that doesn't parse teaches nothing.
`x-understudy-group: G` on a request makes its examples' CV group `G` (id `G#n`), so cross-validation never splits one game.

**Capture log** (`--capture`, default `understudy-capture.jsonl`; JSONL, one line per request, plus `kind:"audit"` lines; the helper threads write them after the responses go out, so their order is not the requests' order):
`ts`, `kind`, `model`, `upstream`, `mode`, `routed` (`local`/`lead`), `status`, `latency_ms` (to the response), `upstream_ms`,
`usage` (the upstream's, zeros for local), `request_bytes`, `state_sha`, `learning` (`learned`, `dropped`, or why not),
`refits`, and `questions`: per question `name`, `spec_hash`, `type`, `opaque`, `lead` (the lead's option), the understudent's
`understudent` (`label`, `probs`, `eligible`, `novel`, `demoted`, `would_answer_local`, `mode`, `readout`), `agree`, and the
learner's `learner_state` and `examples`. `--capture-bodies` adds the full parsed `request` and `response` (off by default;
use it to build replay datasets, and mind what the states contain).

In `race` mode each request line adds `race_id`, `local_ms` (the local computation's time, null when the lead answered
before it finished) and `lead_ms` (the lead call's time; null on a local line, whose lead call is on the audit line).
When a local answer was returned, a second line, `kind: "audit"` with the same `race_id`, `routed: "local"`, `lead_ms`, the
lead's `status`/`usage` and per question `served`, `lead` and **`agree`**, follows once the lead has answered. A request
past `--race-max-inflight` carries `race_capped: true`.

**Overhead** (shadow vs a direct call to a local fake upstream with a fixed 100 ms latency, 1000 calls per side, 8-option
request): added p50 0.27 to 0.48 ms, p95 1.2 to 1.3 ms, p99 1.4 to 2.1 ms, no differing statuses or bytes
(`UNDERSTUDY_OVERHEAD_N=1000 cargo test --release --test serve shadow_adds -- --nocapture`).

**Clients.** A client that appends `/systemone` to its base URL takes `http://127.0.0.1:8787/v1`; one that takes a complete
endpoint takes `http://127.0.0.1:8787/v1/systemone`; one that takes an origin takes `http://127.0.0.1:8787`. The
upstream keys are the leads' (`key_env`, `--route-key-env`), not the caller's. See [docs/api.md](docs/api.md).

### Snapshot format (`understudy-snapshot/1`)

```
DIR/snapshot.json                 format, encoder fingerprint, featurizer (layers, attn layer, pair), settings, lead,
                               audit RNG state, sequence counter; per question: spec, key, fit metadata (readout,
                               mode, threshold, novelty floor, CV report), audit window, demoted, since_refit, and
                               the example window (seq, kind, state hash, label, lead probs, encoding key)
DIR/fits/<key>.safetensors     pool.{mu,sd,coef,intercept}, attn.{mu,sd,q,w,b}, sparse.{coef,intercept} (keywords, blend), neighbors.{vecs,y},
                               blend.{shape_mu,shape_sd,...} (blend's standardizers), memo.{sig,exact,y,conf}, refs (novelty
                               vectors); a ladder's tiers (and an auto race's candidates) are fits/<key>.<tier>.safetensors
DIR/states/<enc>.safetensors   one per encoded state: part<i>.pool (F32), part<i>.tokens (BF16 [t, 512]), for keywords tiers
                               sparse.idx (U32), sparse.val (F32), shape.v (F32) for shape and blend, memo.key (U32) for memo,
                               and a rules input as metadata; a ladder also mean<i>.pool and stat<i>.pool (`probe`, `lite`)
DIR/rules_llm.jsonl            the rules understudent's LLM request/response cache (keyed by sha256 of the request body)
```
Encodings are stored so refits never re-encode; unreferenced ones are deleted on save. A snapshot refuses an encoder
with a different fingerprint (checkpoint revision, precision).

### Seeding and judging

`train` on the first N stream items then `judge` on the rest gives exactly `replay`'s decisions for those items
when the understudent hasn't graduated within the first N items in replay and N is a multiple of 100 (so both have the
same examples, refit at the same point, and an untouched audit RNG). Otherwise they differ by design: replay only
learns from forwarded and audited items, while train stores every item and refits once at the end (`train` doesn't
follow the every-100 refit schedule).

`understudy-bench run --mode split --train N` exercises both: the first N items train, the rest stream through judge.

## understudy-bench (Go/Cobra, in bench/)

- `understudy-bench synth [--datasets ...] [--n 600]`: builds each dataset with an open-weights model
  behind an OpenAI-compatible chat-completions endpoint (the model is recorded in each `synth_stats.json`; see
  [data/README.md](data/README.md)), some programmatically. Resumable; raw generations cached. The endpoint defaults to Fireworks'
  public API (`https://api.fireworks.ai/inference/v1/chat/completions`) with the key in `$FIREWORKS_API_KEY`; `--url`,
  `$UNDERSTUDY_LLM_URL` or `[bench] llm_url` in the lead config file override the URL.
- `understudy-bench teach --lead NAME [--url --model] [--workers 4] [--data-dir data]`: lead answers, cached and resumable.
  NAME is a System One lead of the lead config file (`[leads.NAME]`: `url`, `model`, `key_env`; the Go side reads the same file
  with a minimal TOML reader); `--url` and `--model` together need no config. `run --lead NAME` only names the recorded
  `lead.NAME.jsonl` to replay and needs no config.
- `understudy-bench run --systems understudent,hunch [--datasets ...] --run NAME`: builds streams, runs the replays (`hunch` is an optional external baseline).
  - `--train-ladder 200,1000,2000,5000,10000 [--train-pool <test>=<pool> ...]`: a **fixed-test learning curve** instead.
    For each rung N and each understudent system: `understudy train` on the first N lead-labeled items of the dataset's
    training pool (default `<x>_per` -> `<x>_big_per`; the pool must ask the same question), then `understudy judge
    --frozen --replay-lead` over the dataset's own items, which are therefore the same held-out set at every rung.
    `hunch` is skipped (it learns from every request). Output per dataset: `ladder.json` (per rung: train size and groups,
    the pool's majority share, grouped-CV agreement/coverage/minority recall and the CV Wilson lower bound, held-out
    agreement of the understudent's own label with the lead, held-out majority share, lift, held-out Wilson lower bound
    and whether it clears 0.90, coverage and local agreement), `decisions.<system>-n<N>.jsonl`, `train.<system>-n<N>.json`.
    Rungs beyond the pool's lead-labeled prefix are skipped, not failed. `train` fits once per rung, so the refit
    schedule (every 100 items) plays no part in a ladder; replay on a 10k stream is where it would.
    Each rung passes `train --max-examples N` (ordinary `train` keeps only the newest 2000 sampled examples, evicting
    the oldest, so without it a 5,000 rung would fit items 3,001-5,000), and **fails** if the fit used fewer than N
    minus the duplicate states among the first N (both reported per rung as `train_dedupe_drops` and `fit_n`); the CV
    Wilson bound uses `fit_n`. The pool must be disjoint from the test set: a shared id or a prompt equal after
    normalization rejects the run, and a pool prompt that is a near-duplicate of a test prompt (4-gram Jaccard >= 0.8,
    word-set Jaccard >= 0.7 or edit similarity >= 0.85) is excluded with all its sub-items, without renumbering
    (`ladder.json`: `pool_excluded_parents`). A pool is built outside the harness (per-response items with ids `<item>#<k>`,
    answered by the same lead, whose `state_sha` must match the item's current state).
    Each rung also carries a **length shortcut** diagnostic (`ladder.json` `length_shortcut`, a table in `LADDER.md`;
    lengths only, so it is text-free) for datasets whose states have a `response`: the lead's yes-rate by response-length
    quartile on the pool prefix and on the test (quartile cut points are the test's), the understudent's yes-rate by quartile
    on the test, and the Spearman correlation of p(yes) with length on the test for the understudent and for the lead; the
    single number to watch is **`spearman_delta` = understudent minus lead** (large and positive: the understudent leans on
    length more than the lead does).
- `--systems` also takes `understudent:shape`, `understudent:blend`, `understudent:memo`, `understudent:net`, `understudent:checklist` (and `understudent:auto`, `understudent:ladder`;
  pass `--understudent-arg --candidates --understudent-arg keywords,lite,probe,deep,shape,blend,memo` for the extended race).
- `understudy-bench report --run NAME`: metrics and `runs/<run>/REPORT.md` + `results.json`; both carry a **Usefulness**
  section, per dataset x system: coverage; agreement with the lead on the answered share and its Wilson lower bound;
  the understudent's own latency p50/p99 (`understudent_us`) on the items it answered, and on the items it forwarded (its
  overhead in front of the lead); the lead's latency p50/p99 (`latency_ms` of the lead answers); and **$ saved per 1M
  calls** = coverage x lead cost per call x 1,000,000 (JSON `saved_per_1m_calls_usd`; `saved_per_1k_calls_usd` is kept alongside it, = coverage x lead cost per call x 1000). The lead cost per call is (state bytes / 3.4 + 600) tokens x $0.042 per
  million uncached input tokens, output free, for both Jev and Rune (`report.LeadRates`, the one place the rates live;
  source: the provider's billing API, 2026-11-01, and list price), averaged over the stream's states; a lead without a
  measured rate shows no dollars; for a ladder run,
  `results.json` + `LADDER.md` (the learning-curve tables).
