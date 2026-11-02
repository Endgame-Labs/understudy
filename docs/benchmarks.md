# Benchmarks

How Understudy is measured, what the numbers mean, and where to find them. Every number here comes from the committed
replays in `runs/`, over the eleven synthetic datasets in [`data/`](../data/README.md).

- [Method](#method)
- [Metrics](#metrics)
- [Cost at scale](#cost-at-scale)
- [Results](#results)
- [Known limits](#limitations)
- [Reproducing](#reproducing)

## Method

**Replay.** `understudy-bench` records each dataset's lead answers once (`teach`), then replays the items through an understudent
**deterministically and offline**. Items are processed one at a time in stream order. For each one the understudent predicts
with its current fit. If it may answer (graduated, confident, not novel, not demoted) it answers, and with probability
`audit_rate` (seeded, seed 0) the lead's recorded answer is also used as an audit and as a new example. Otherwise the item is
forwarded: the caller gets the lead's recorded answer, the example is stored, and every understudent that *would* have
answered records an audit. A refit runs synchronously every 100 new examples, so a run repeats exactly.

**Cold start.** The understudent starts empty and learns from the stream itself, exactly as it would in front of live
traffic. That is the realistic number: coverage includes the time spent learning. A **warm start** (`--mode split`) seeds it with
the first N lead answers via `train` and judges the rest.

**Fixed-test ladder.** To ask whether more labeled data helps, `--train-ladder` trains on the first N items of a disjoint pool
and scores a *frozen* judge on the same held-out items at every rung. See [examples](examples.md#train-ladder).

**Question types.** The eleven datasets cover narrow gates and classifiers of different decision kinds (surface property, topic or
kind, graded tone, a needle in long text, a minimal pair, a pair comparison, relational match, binding and lookup, a conditional
rule, world knowledge, a compound judgment). The expected outcome differs by kind: some should take over, some should
correctly abstain. Each dataset's items are generated with the label known by construction, and the lead's recorded answers
are what the understudent learns from.

## Metrics

Reported per dataset and system in the **Usefulness** section of `runs/<run>/REPORT.md` and `results.json`.

| metric | definition |
|---|---|
| **coverage** | share of requests answered locally |
| **agreement** | on the answered share, how often the local answer equals the lead's |
| **Wilson low** | the 95% Wilson lower bound of that agreement, with `n` = the answered count. The number to trust: a high agreement over a handful of answers has a low bound |
| **latency p50 / p99** | the understudent's own time (`understudent_us`) on the items it answered, and separately on the items it forwarded (its overhead in front of the lead) |
| **lead latency p50 / p99** | `latency_ms` of the lead's recorded answers |
| **$ saved per 1M calls** | coverage × lead cost per call × 1,000,000 (savings scale linearly with volume; see [Cost at scale](#cost-at-scale)) |

Where a verdict is needed, **takes over** means 2nd-half coverage of at least 50% with accuracy within 1 point of the lead, **harmful**
means graduated with accuracy more than 2 points below the lead, and **correctly abstains** means never graduated.

**Lead cost per call** is `(state bytes / 3.4 + 600)` input tokens × **$0.042 per million uncached input tokens**, output free,
for both Jev and Rune (the rates live in one place, `report.LeadRates`; measured from the provider's billing for one and
list price for the other). The estimate is averaged over the stream's states. A lead without a measured rate shows no
dollars.

## Cost at scale

Every call answered locally is a lead call you never pay for. Savings grow linearly with call volume, and the understudent's
own cost is CPU time, which costs far less than per-token lead calls. A data pipeline that makes millions to billions of
lead calls saves real money; latency and availability are co-equal wins on top.

A call whose state is 3,000 bytes is about `3,000 / 3.4 + 600 = 1,482` input tokens (the per-call model above). **Net local
share** is coverage minus the audits (by default one local answer in ten also goes to the lead), so 80% coverage is about
72% net. Output tokens, which larger models charge for, would only raise these numbers.

Lead dollars saved:

| lead input price per million tokens | net local share | 1M calls | 100M calls | 1B calls |
|---|---:|---:|---:|---:|
| $0.042 (measured Jev / Rune rate) | 25% | $16 | $1,556 | $15,565 |
| $0.042 (measured Jev / Rune rate) | 50% | $31 | $3,113 | $31,129 |
| $0.042 (measured Jev / Rune rate) | 75% | $47 | $4,669 | $46,694 |
| $0.20 (typical small hosted model) | 25% | $74 | $7,412 | $74,118 |
| $0.20 (typical small hosted model) | 50% | $148 | $14,824 | $148,235 |
| $0.20 (typical small hosted model) | 75% | $222 | $22,235 | $222,353 |
| $1.00 (typical mid-size model) | 25% | $371 | $37,059 | $370,588 |
| $1.00 (typical mid-size model) | 50% | $741 | $74,118 | $741,176 |
| $1.00 (typical mid-size model) | 75% | $1,112 | $111,176 | $1,111,765 |
| $3.00 (typical large model) | 25% | $1,112 | $111,176 | $1,111,765 |
| $3.00 (typical large model) | 50% | $2,224 | $222,353 | $2,223,529 |
| $3.00 (typical large model) | 75% | $3,335 | $333,529 | $3,335,294 |

CPU cost of the same work, charged on **every** call (the understudent computes features for each call it sees, answered or
forwarded), at an assumed $0.04 per vCPU-hour:

| design | CPU time per call | 1M calls | 100M calls | 1B calls |
|---|---:|---:|---:|---:|
| cheap designs (`keywords`, `shape`, `checklist`, `lite`) | ~0.2 ms | $0.002 | $0.22 | $2.22 |
| encoder designs (`probe`, `deep`, `net`) | ~75 ms | $0.83 | $83 | $833 |

At the measured rate and 50% net, a billion calls save about $31,000 for about $2 of CPU with a cheap design (or about $830
with an encoder design). At larger-model rates the same billion calls save from about $150,000 to over $2 million.

Latency and availability matter as much:

| | per call |
|---|---|
| a lead call | hundreds of milliseconds to seconds (queueing, network, a model) |
| a local answer from `keywords` or `shape` | well under a millisecond |
| a local answer from `probe` or `deep` | tens of milliseconds (`deep` 66 to 89 ms at p50) |
| a cheap design's own time on a call it forwards | 0.06 to 0.23 ms at p50 (see [Latency](#latency)) |

A graduated question keeps answering when the lead is slow, rate-limited or down (only its audits wait). The tables below
report coverage, agreement, latency and dollars side by side.

## Results

Replay benchmark, October and November 2026: the 11 synthetic questions in `data/`, each with **both leads** (Jev and Rune),
10 designs racing, cold start, graduation target 0.90 (`runs/narrow-v1-jev`, `runs/narrow-v1-rune`; `runs/latencyd-jev` for
clean latency, `runs/pf-*-data-jev` for the `auto` speed preference). Earlier revisions of the same sets are `runs/v0` (Rune,
including warm starts) and `runs/v1-target97` (Rune, target 0.97). Runs are committed text-free.

**Coverage** is the share of calls answered locally over the whole cold-start stream (learning included). **Agreement** is
agreement with the lead on that answered share. The best design is the highest-coverage design whose realized agreement
stayed at or above the target; `auto` races the others and serves one.

| question | type | n | Jev: coverage @ agreement | Rune: coverage @ agreement | best design | design latency (p50, CPU) | verdict |
|---|---|---:|---:|---:|---|---|---|
| politer_reply | pairwise preference | 600 | **80%** @ 94.2% | **80%** @ 94.2% | keywords | ~0.1-0.2 ms | helps |
| payment_flip | yes/no gate | 600 | **69%** @ 91.5% | **70%** @ 92.3% | deep | ~65-90 ms | helps |
| output_kind | classify tool output | 600 | **58%** @ 97.7% | **64%** @ 97.9% | deep | ~65-90 ms | helps |
| retry_exception | conditional yes/no gate | 600 | **53%** @ 93.4% | **52%** @ 97.1% | auto, deep | see auto, ~65-90 ms | helps |
| tool_failed | error detection (yes/no) | 600 | **25%** @ 98.0% | **39%** @ 97.4% | deep, `probe` | ~65-90 ms | helps |
| passage_answers | relevance (yes/no) | 600 | **14%** @ 97.7% | **16%** @ 94.7% | deep | ~65-90 ms | partial |
| incident_severity | triage (score) | 600 | **11%** @ 91.2% | 0% | deep | ~65-90 ms | partial |
| capital_city | lookup (yes/no) | 600 | 0% | 0% | none | n/a | lead-only |
| code_lookup | lookup (choice) | 600 | 0% | 0% | none | n/a | lead-only |
| customer_frustration | sentiment (score) | 600 | 0% | 0% | none | n/a | lead-only |
| secret_needle | needle detection | 600 | 0% | 0% | none | n/a | lead-only |

Lead-only questions are harmless: nothing graduated, so every call was forwarded. The mean time `auto` spent per call on the
four lead-only sets, learning included, was 0.07 ms (capital_city), 0.26 ms (code_lookup), 0.50 ms (customer_frustration) and
2.9 ms (secret_needle, whose long logs are the costliest to featurize).

### Every design, Jev as the lead

Coverage @ realized agreement per design (0% means it never graduated):

| dataset | keywords | lite | probe | deep | neighbors | shape | blend | memo | auto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| capital_city | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| code_lookup | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| customer_frustration | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| incident_severity | 3% @ 100.0% | 0% | 0% | 11% @ 91.2% | 0% | 0% | 0% | 0% | 11% @ 91.2% |
| output_kind | 52% @ 98.1% | 51% @ 97.7% | 54% @ 96.9% | 58% @ 97.7% | 24% @ 98.6% | 0% | 55% @ 97.6% | 0% | 55% @ 98.5% |
| passage_answers | 0% | 0% | 0% | 14% @ 97.7% | 0% | 0% | 0% | 0% | 14% @ 97.7% |
| payment_flip | 26% @ 99.4% | 23% @ 100.0% | 50% @ 89.6% | 69% @ 91.5% | 0% | 0% | 18% @ 100.0% | 0% | 69% @ 91.5% |
| politer_reply | 80% @ 94.2% | 73% @ 97.5% | 78% @ 99.6% | 78% @ 98.7% | 64% @ 95.8% | 58% @ 88.9% | 76% @ 96.3% | 0% | 80% @ 94.2% |
| retry_exception | 27% @ 98.1% | 12% @ 94.5% | 53% @ 93.4% | 51% @ 97.0% | 0% | 0% | 16% @ 95.7% | 0% | 53% @ 93.4% |
| secret_needle | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| tool_failed | 12% @ 98.6% | 0% | 22% @ 96.9% | 25% @ 98.0% | 0% | 0% | 0% | 0% | 25% @ 98.0% |

### Every design, Rune as the lead

| dataset | keywords | lite | probe | deep | neighbors | shape | blend | memo | auto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| capital_city | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| code_lookup | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| customer_frustration | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| incident_severity | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| output_kind | 51% @ 98.0% | 52% @ 95.2% | 57% @ 98.0% | 64% @ 97.9% | 44% @ 97.7% | 0% | 51% @ 98.7% | 0% | 64% @ 97.9% |
| passage_answers | 0% | 0% | 0% | 16% @ 94.7% | 0% | 0% | 0% | 0% | 16% @ 94.7% |
| payment_flip | 14% @ 98.8% | 25% @ 100.0% | 47% @ 89.4% | 70% @ 92.3% | 0% | 0% | 20% @ 100.0% | 0% | 70% @ 92.3% |
| politer_reply | 80% @ 94.2% | 73% @ 97.3% | 78% @ 99.8% | 78% @ 98.3% | 64% @ 95.8% | 58% @ 90.3% | 76% @ 96.3% | 0% | 80% @ 94.2% |
| retry_exception | 27% @ 98.8% | 13% @ 94.7% | 51% @ 94.4% | 52% @ 97.1% | 0% | 0% | 15% @ 95.6% | 0% | 51% @ 94.8% |
| secret_needle | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| tool_failed | 32% @ 98.5% | 0% | 39% @ 97.4% | 28% @ 97.6% | 0% | 0% | 0% | 0% | 39% @ 97.4% |

Both leads agree within a few points on every question where something graduates: the cheap-versus-encoder ordering is the
same, and the questions that stay lead-only are the same ones (except incident_severity, where Rune's answers were noisier and
nothing cleared the bar).

### A stricter target

With `--target 0.97` (`runs/v1-target97`, Rune, an earlier code revision), coverage drops where the margin was thin, and every
graduated design realized at least 95.4% agreement:

| dataset | keywords | lite | probe | deep | neighbors | ladder |
|---|---:|---:|---:|---:|---:|---:|
| capital_city | 0% | 0% | 0% | 0% | 0% | 0% |
| code_lookup | 0% | 0% | 0% | 0% | 0% | 0% |
| customer_frustration | 0% | 0% | 0% | 0% | 0% | 0% |
| incident_severity | 0% | 0% | 0% | 0% | 0% | 0% |
| output_kind | 20% @ 98.3% | 22% @ 98.5% | 22% @ 100.0% | 27% @ 98.8% | 10% @ 100.0% | 29% @ 97.7% |
| passage_answers | 0% | 0% | 0% | 0% | 0% | 0% |
| payment_flip | 0% | 0% | 21% @ 98.4% | 34% @ 100.0% | 0% | 34% @ 100.0% |
| politer_reply | 29% @ 100.0% | 45% @ 100.0% | 64% @ 100.0% | 64% @ 100.0% | 44% @ 99.6% | 64% @ 100.0% |
| retry_exception | 16% @ 99.0% | 0% | 33% @ 95.4% | 26% @ 98.1% | 0% | 27% @ 98.1% |
| secret_needle | 0% | 0% | 0% | 0% | 0% | 0% |
| tool_failed | 32% @ 98.5% | 0% | 39% @ 97.4% | 0% | 0% | 32% @ 98.5% |

The `ladder` column is the default tiers answering cheapest first. A stricter target costs coverage on the questions that
barely cleared 0.90 (payment_flip and politer_reply for `keywords`, the whole of passage_answers) and little on the clean
ones (tool_failed `keywords` is unchanged at 32%).

### Latency

Idle machine, no encode cache, Jev answers (`runs/latencyd-jev`, ranges over politer_reply and output_kind):

| design | answers locally, p50 / p99 | its own time on a forwarded call, p50 / p99 |
|---|---|---|
| shape | 0.07 / 0.15 ms (politer_reply only) | 0.06-0.07 / 0.22-0.25 ms |
| keywords | 0.10-0.21 / 0.5-4.8 ms | 0.07-0.15 / 0.4-0.5 ms |
| checklist | 0.15-0.24 / 0.6-0.7 ms | 0.10-0.19 / 0.5-0.6 ms |
| lite | 0.19-0.22 / 0.5-1.6 ms | 0.20-0.23 / 0.5-4.3 ms |
| deep | 66-89 / 172-476 ms | 60-72 / 126-534 ms |
| `auto` (default candidates) | 0.23 / 0.6 ms on politer_reply (keywords), 44 / 537 ms on output_kind (a `deep` candidate) | 75-88 / 174-714 ms |
| **lead** (Jev, measured) | **78-82 / 152-204 ms** | n/a |

The cheap designs answer in about a tenth of a millisecond. `deep` runs a transformer per request on CPU and costs about as
much time as the lead itself; worse, an `auto` race that includes an encoder candidate spends that time on every forwarded call,
because each forwarded item becomes a training example for every candidate. Two consequences: prefer a cheap design when its
coverage is close (`auto --prefer-fast`, below), and use `--mode race` (send to the lead immediately, answer locally if
eligible) when an encoder design serves.

### Preferring the cheaper design

`auto --prefer-fast EPS` lets a cheaper candidate serve when its coverage is within EPS of the best
([auto](understudents/auto.md)). On these three questions at 0.05 the serving choice is unchanged and the latency
differences are run-to-run noise; at 0.12 tool_failed serves `probe` (answered p50 52 ms against 70 ms, 21.8% coverage
against 25.3%, agreement 96.9% against 98.0%). The numbers are in the [auto](understudents/auto.md) page
(`runs/pf-off-data-jev`, `runs/pf-on-data-jev`, `runs/pf12-data-jev`).

### Dollars

Coverage turns straight into lead calls not made. At the measured lead rate (about $0.00006 for a 3,000-byte call), 80%
coverage (about 72% net of audits) saves about **$45 per million calls, $4,500 per 100 million and $45,000 per billion**,
for about $2 of CPU with the cheap designs (about $830 with the encoder designs). At larger-model prices multiply by 5 to 70. See
[Cost at scale](#cost-at-scale).

### Scope

| helps | does not help |
|---|---|
| **Narrow classification and judging with a visible signal**: pairwise preference (80% of calls at 94% agreement, in 0.2 ms), classifying tool output (58-64% at 98%), yes/no gates (25-70%) | **Broad single-question decisions over huge state spaces** stay lead-only: every state makes a new question, nothing ever graduates. They cost only the forwarding overhead |
| Questions asked thousands of times about similar-looking states | **Decisions that compare two changing texts** (a fresh query against a fresh passage: search relevance, reranking): passage_answers reaches only 14-16%. These are lead-only for these small models |
| Both leads alike: the results agree within a few points | **Skewed questions** (the lead gives one answer ~99% of the time): Understudy refuses to certify the rare answer, which is the one that matters |
| | **Lookups, graded sentiment and needles in long text** (capital_city, code_lookup, customer_frustration, secret_needle): nothing graduates, correctly |
| | **Small streams** (under ~100 labeled calls) and **fuzzy questions** where no design beats the majority class |

## Limitations

- **Synthetic data only.** Every dataset here was generated with the label known by construction, 600 items each. The numbers
  show which kinds of question a design can learn, not what your traffic will do; run `understudy serve --mode shadow` first
  and read the capture log.
- **Two leads.** Only Jev and Rune were benchmarked as leads. Another lead's answers may be easier or harder to learn.
- **Cold start.** Coverage includes the time spent learning, so it grows with stream length; a 600-item stream understates a
  long-running one (the second-half coverage in the run reports is typically 15 to 25 points higher).
- **Realized agreement can dip under the target.** A few designs graduated and then realized agreement just under 0.90 on the
  rest of the stream: Jev payment_flip `probe` 89.6%, and politer_reply `shape` 88.9% (2 demotions). A pointwise certificate
  re-tested at every refit is optimistic. Audits and demotion are the backstop, and `--graduation sequential` removes the effect
  at some cost in coverage ([graduation](graduation.md)).
- **Encoder designs cost real time.** `deep` and `probe` take tens of milliseconds on CPU, close to a lead call; use `--mode
  race` when one serves, or let `auto --prefer-fast` choose the cheaper design.
- **Rare classes.** A rare answer is certified only with a minority-class agreement bar, so a heavily skewed question usually
  stays lead-only.
- **Platforms.** Unix only. SIMD is runtime-dispatched on x86_64 and aarch64.
- **Studies not reported here.** Broader single-question decisions over huge state spaces, best-of-N response judging and
  other fuzzy or skewed real-world gates were also tried and stayed lead-only; the datasets are not part of this repo, so no
  numbers are reported.

## Reproducing

```bash
cargo build --release
(cd bench && go build -o ../target/understudy-bench .)

target/understudy-bench run --run mybench --lead jev --systems understudent:keywords,understudent:deep,understudent:auto
target/understudy-bench report --run mybench          # runs/mybench/REPORT.md, results.json
```

`--data-dir DIR` points at another dataset directory (the default is `data/`). A run over datasets whose text must not be
committed deletes its text artifacts after the replay and verifies that nothing left carries state text, so committed `runs/`
stay text-free. Keep `cargo test --release` and `(cd bench && go vet ./... && go test ./...)` green.
