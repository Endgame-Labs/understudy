# TODO

Open work. Results are in [docs/benchmarks.md](docs/benchmarks.md).

## Benchmarks

- [ ] Benchmark `rules` on the narrow datasets (both leads) before claiming where it helps; it was never in the 10- or
  6-design runs.
- [ ] `understudy-bench repro public`: re-run the synthetic narrow sets from the committed lead answers and check coverage and
  agreement against the shipped `results.json` within a tolerance.

## Next designs and features

- [ ] A small distilled transformer: a compact encoder trained on the lead's answers, to close the gap between the cheap
  designs (~0.1 ms) and the encoder designs (tens of milliseconds).
- [ ] An option-aware design that reads the candidate options themselves (for choice questions whose options change per
  call), not only the state.
- [ ] Per-question target recommendations: suggest a graduation target per question from its replay (coverage against
  agreement), instead of one global `--target`.
- [ ] A canonicalization cross-check between the Go bench and the Rust server: the same request must produce the same
  question identity and features in both.

## Preview release (v0.1.0-preview.1)

The release is a fresh-history public cut-over done by the maintainer: this tree, with its history replaced, becomes the
public repository under the MIT license. Nothing is published by automation.

- [x] Leads: `chat` adapter for OpenAI-compatible endpoints, `--lead-kind`, `--lead-key-env`, presets from a config file
  (no internal hostnames in code), public defaults in the bench generators
- [x] Docs at the source: benchmarks over the committed synthetic data only, known limits, [security](docs/security.md)
- [ ] Model fetch: `understudy model fetch` (pinned revision, SHA-256 per file), revision pin in the lookup, fixed error hint
- [ ] `docs/getting-started.md`: under 5 minutes, model-free designs on the committed synthetic data, tested from a clean
  container
- [ ] CHANGELOG.md, NOTICE (vendored code, the embedding model and its license), version `0.1.0-preview.1`
- [ ] CI (`ci.yml`: tests, macOS build, MSRV, getting-started, link check) and `release.yml` (four targets, SHA256SUMS, draft only)
- [ ] Capture and snapshot files created with owner-only permissions (today they follow the umask)
- [ ] The maintainer: may the Jev and Rune recorded answers in `data/` and `runs/` ship, or are they re-labeled with a
  public model first?

## After the release

- [ ] **Fix: encoder options without weights.** `auto` with `--layers`, `--attn-layer` or `--readout` and no encoder weights fails validation (`src/main.rs` ~585-599): the namespace stores a permanent error and never reaches the upgrade. Validate against the full candidate set and keep the options for the encoder tiers. The default (no encoder options) is not affected.

- [ ] **Lead adapters** (see [docs/leads](docs/leads/README.md)).
  - Cloudflare Clef / Clef Flash: unwrap the `{"result": ..., "success", "errors"}` envelope, and map a failure to an HTTP error. Everything else (path, Bearer token, body) already works through `--route`.
  - OpenAI Decisions (and Fireworks' `/decisions`): translate System One to `input` + ordered `questions` (`predicate`, `choice` with `choices`, `score` with `levels`) and the array answers back.
  - Jev Box and image answers through `serve`: verify the header and `assets` pass-through live.
- [ ] **Pass-through of unknown paths** (streaming-safe, origin-form targets only, authority pinned, bounded state).
- [ ] **Live challengers in `auto`.**
  - Keep the runners-up that are statistically tied with the champion as live challengers, using a paired test on the same audited items. Cap them at 3 by default (configurable), cheapest first.
  - Score challengers only where a lead answer exists (forwarded and audited items), so the cost scales with lead calls. Encoder designs share one encode.
  - Promote a challenger only when the sequential test finds it significantly better. On demotion, the best passing challenger takes over at once, with no fallback to the lead.
  - Benchmark it on drifting streams.
- [ ] **Image understudents.** An image encoder (SigLIP2 or CLIP).
  - Start with a yes/no question ("is this a hot dog?").
  - Then box localization, with IoU-based agreement, on COCO / Open Images hot dog and banana.
  - Box answers are a Jev-only capability: Jev teaches and audits.
- [ ] **A router for the design.** The routing question is itself a classification: return type (yes/no, choice, score), kind of
  question (surface property, topic, graded tone, needle, minimal pair, pair comparison, relational match, lookup, conditional
  rule, world knowledge, compound judgment) and state shape (short text, long text, two fields, record list, structured JSON).
  Write the empirical table "kind -> best design, or don't try" from the benchmark, see whether the kind can be inferred
  cheaply from the spec and a few states, and declare lead-only up front for kinds that never graduate, saving the warm-up.
- [ ] **Statistics upgrades.**
  - Anytime-valid graduation (confidence sequences) as the default instead of re-testing a fixed 95% bound at every refit;
    measure the difference in false graduations. `--graduation sequential` is the opt-in version today.
  - Lead version in the question key: a lead change (a new model version, or one lead replaced by another) should restart
    graduation checks instead of waiting for audits to notice.
  - Adaptive audit rate: audit more right after graduation or on drift, less once stable.
  - Soft targets: train on the lead's probabilities instead of the argmax weighted by confidence.
- [ ] **A lead ladder.** Understudy answers, then a cheap lead, then a stronger one, each tier graduating against the next.
- [ ] **Research questions.**
  - Which lead is easiest to distill? A lead whose answers a frozen-embedding probe learns quickly may be leaning on surface
    features.
  - Does the learned model beat the lead on consistency (order effects, repeat stability)?
  - Real traffic instead of synthetic: public agent traces and public tool-output datasets.
- [ ] **More lead-answer data for the near-misses.** A train-size ladder (200 to 10,000 lead labels, scored on a fixed held-out
  set, grouped CV inside the pool) on questions whose learning curve is still rising. Build the pool from real, length-matched
  responses, since a pool of short templated ones teaches length shortcuts; the ladder's length-shortcut diagnostic shows
  whether a result leans on length.
- [ ] **Build speed.**
  - Share one `CARGO_TARGET_DIR` across worktrees, so parallel work doesn't cold-build.
  - Fold `tests/*.rs` into one integration-test crate (`tests/it/main.rs` with modules), so the library links once instead of 13 times.
  - Try sccache.
  - Measure clean and incremental `cargo test` times before and after, and record them here.
