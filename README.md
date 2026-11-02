# understudy

Understudy is a drop-in, CPU-only proxy for System One models such as Jev, Rune, and Clef (the "lead" models).

System One models are cheap and fast enough that they spread into infrastructure: data pipelines, ETL, MapReduce jobs, event reactors, and Kafka consumers. At that volume, the same narrow questions get asked millions of times, and the lead bill grows with them (Jevons paradox). Understudy is for that situation. If your System One volume is small, you don't need it.

Understudy forwards each request to the lead and trains small CPU models, *understudents*, on the lead's answers: an extremely narrow distillation of "What Would Jev Do?". Several understudent designs compete on each type of request. When one reaches near-parity with the lead, it graduates, and Understudy answers that question itself instead of calling the lead.

It never stops auditing graduated understudents against the lead. Requests drift, outliers appear, and cardinality grows, so any understudent that slips is demoted and the question goes back to the lead.

```mermaid
flowchart TD
    C["caller"] -->|"request"| U(("understudy"))
    U -->|"graduated"| S["Student answers"]
    U -->|"not graduated"| L["Lead answers<br/>(Jev/Rune/etc.)"]
    S --> R["response"]
    L --> R
    R -->|"same API, same shape"| C
    L -.->|"examples"| T["understudents<br/>(design race per question)"]
    T -.->|"graduation"| U
```

Most understudent designs are a linear model over one of three inputs: hashed word and character n-grams, structural
features of the text, or EmbeddingGemma 2 vectors. There are 11 designs ([docs/understudents](docs/understudents/README.md)).

## Production Loop

A request is answered by its understudent only when the question has graduated ([graduation statistics](docs/graduation.md)).
Otherwise, the lead answers, and the request becomes a training example. Sampled audits compare the understudent with the
lead, and bad audits demote it.

```mermaid
flowchart TD
    R["live request"] --> G{"graduated, confident,<br/>not novel?"}
    G -- "yes" --> S["understudent answers"]
    G -- "no" --> L["lead model answers<br/>(returned to the caller)"]
    L --> X["store the example"]
    S --> AU["sampled audit:<br/>student vs lead"]
    AU -- "agree" --> G
    AU -- "bad audits" --> D["demote: back to the lead"]
```

## Learning/Graduation Loop

In `auto` mode, every candidate design is refit on each question's examples, and only one that clears the bar may serve.

```mermaid
flowchart TD
    E["examples accumulate<br/>per question"] --> F["each refit: fit every candidate<br/>(keywords, shape, checklist, memo,<br/>deep, probe, ...)<br/>with grouped out-of-fold CV"]
    F --> B{"clears the bar?<br/>Wilson lower bound<br/>+ minority-class agreement"}
    B -- "none" --> E
    B -- "one or more" --> P["auto picks the best<br/>(the cheaper one when close)"]
    P --> SV["it serves"]
    SV -- "audits demote it" --> E
```

## Scope

Understudy targets narrow, repetitive decisions: yes/no gates, classification, routing, triage, moderation, relevance,
and judging. In questions such as "Is this support ticket urgent?", "Does this passage mention a refund?", and "Is this
tool output an error?", the question is fixed and only the content varies. A small model can learn these from a few
hundred lead answers.

Understudy does not help with broad single-question decisions over large state spaces, such as choosing a move in a game.
Those questions stay on the lead and never graduate; they cost only the forwarding overhead (about 0.3 ms per call, with
no extra lead spend). Decisions that compare two changing texts (a fresh query against a fresh passage, as in search
relevance or reranking) are also lead-only for these small models. Because nothing is answered locally until it
graduates, a question that cannot be learned stays on the lead.

## Setup

1. **Install**

   Pick one channel. [docs/install.md](docs/install.md) has the details and checksum verification.

   - Homebrew: `brew install endgame-labs/oss/understudy`
   - Shell installer: `curl -fsSL https://github.com/endgame-labs/understudy/releases/download/v0.1.0-preview.2/understudy-installer.sh | sh`
   - Debian and Ubuntu: `curl -fsSLO https://github.com/endgame-labs/understudy/releases/download/v0.1.0-preview.2/understudy_0.1.0.preview.2_amd64.deb && sudo apt install ./understudy_0.1.0.preview.2_amd64.deb`
   - Docker: `docker run -p 8787:8787 -v understudy-data:/data ghcr.io/endgame-labs/understudy:preview --lead-url <URL>`
   - From source: `cargo install --git https://github.com/endgame-labs/understudy`

2. **Run**

   ```bash
   understudy serve --lead-url <your lead's /v1/systemone URL>
   ```

   It forwards every request to the lead until a question graduates, then answers that question itself; add `--mode shadow` to watch without answering.

3. **Point your app at it**

   Point your app at it. If you use the TypeSafe SDK, that's one environment variable, no code change (it only affects the TypeSafe SDK):

   ```bash
   export TYPESAFE_BASE_URL=http://127.0.0.1:8787
   ```

   The same works with [Fireworks Rune](docs/leads/fireworks-rune.md), [Cloudflare Clef](docs/leads/cloudflare-clef.md) and [OpenAI's Decisions API](docs/leads/openai-decisions.md) (and [TypeSafe Jev](docs/leads/typesafe-jev.md)).

   Any HTTP client can send the same request with curl:

   ```bash
   curl -s http://127.0.0.1:8787/v1/systemone \
     -H 'content-type: application/json' \
     -d '{
       "model": "jev-latest",
       "questions": {
         "urgent": {
           "type": "noul",
           "instructions": "Is this support ticket urgent?",
           "criteria": {
             "true": "the customer cannot work, loses money, or reports a security problem",
             "false": "a question, a request, or a complaint that can wait"
           }
         }
       },
       "state": {
         "subject": "Cannot log in since this morning",
         "body": "Our whole team is locked out and we have a launch today."
       }
     }'
   ```

### Modes

| `--mode` | Behavior |
|---|---|
| `forward` | Forwards every request to the lead and logs it. Does not train. |
| `shadow` | Watch only: forwards every request to the lead, trains on its answers in the background, and logs the candidate local answer. Never answers locally. |
| `serve` (default) | Forwards every request to the lead until a question graduates, then answers that question locally when it is confident. Demotes it on bad audits. No restart is needed. |
| `race` | Calls the lead and the local model concurrently and returns whichever valid answer finishes first. |

See [docs/benchmarks.md](docs/benchmarks.md) for offline replay benchmarking and [docs/examples.md](docs/examples.md) for end-to-end examples.

## API

Request body: `{"model", "questions": {name: {"type", "instructions", "criteria"}}, "state": any JSON}`. Supported question types are `noul` (yes/no), `choice` (one of several options), and `score` (ordered levels).

Response body: `{"model", "answers": {name: answer}, "usage": {...}}`.

- Forwarded responses return the lead's exact bytes with `x-understudy: lead`. Upstream errors (400, 402, 413, 422, 429, and 503 with `Retry-After`) pass through unchanged.
- Local responses return `x-understudy: local` and all-zero `usage` fields. In `serve` mode, a request is answered locally only if every question in the request qualifies.
- `GET /healthz` and `GET /status` report server health and each question's state (`learning`, `graduated`, `demoted`, `lead_only`).

See [docs/api.md](docs/api.md) for the full contract and `curl` examples.

## Graduation

Each question retains its 2,000 most recent lead answers. Every 100 examples, Understudy refits the model and computes 5-fold out-of-fold predictions. A question graduates at the lowest confidence threshold where the out-of-fold predictions at or above that threshold satisfy both conditions:

- The Wilson lower bound of agreement with the lead is at least 0.90 across at least 30 rows.
- The minority class agrees at a Wilson lower bound of at least 0.80.

One-sided variants allow a yes/no question to graduate on "no" only or "yes" only. After graduation, roughly 1 in 10 local answers is audited against the lead. A Wilson bound over the last 100 audits demotes the question if it has drifted. Input states outside the observed training distribution are forwarded as novel. Passing `--graduation sequential` (opt-in) allocates the error budget across refits so repeated evaluations do not graduate a below-target question by chance. See [docs/graduation.md](docs/graduation.md).

## Benchmarks

Every call answered locally avoids a paid lead call. Savings grow linearly with call volume while the understudent uses only CPU time. At the measured lead rate, 80% coverage (72% net of audits) saves about $45,000 per billion calls against about $2 of CPU time with a lightweight design (about $830 with an encoder design), and 5 to 70 times that amount at larger-model prices ([Cost at scale](docs/benchmarks.md#cost-at-scale)). Local inference also lowers latency (about 0.1 ms versus ~80 ms for the lead) and continues serving graduated questions when the lead is slow or unreachable.

The benchmark suite covers 11 synthetic question types (the datasets in [data/](data/README.md)) across both leads with all model designs competing (replay, cold start, target 0.90). Coverage is the share of calls answered locally. Agreement is accuracy against the lead on those calls.

| question type | Jev | Rune | best design (local latency) |
|---|---:|---:|---|
| pairwise preference | **80%** @ 94.2% | **80%** @ 94.2% | keywords (**~0.2 ms**) |
| yes/no gate | **69%** @ 91.5% | **70%** @ 92.3% | deep (~65-90 ms) |
| classify tool output | **58%** @ 97.7% | **64%** @ 97.9% | deep (~65-90 ms) |
| yes/no gate (retry?) | **53%** @ 93.4% | **52%** @ 97.1% | auto / deep |
| error detection | **25%** @ 98.0% | **39%** @ 97.4% | deep / `probe` |
| passage relevance, triage score | 11-14% | 0-16% | deep, partial |
| lookups, graded sentiment, needles in long text | 0% | 0% | **lead-only** |

Lead latency is ~80-90 ms. The lightweight designs answer in about 0.1 ms and the deep encoder design in about 65-90 ms.

See [docs/benchmarks.md](docs/benchmarks.md) for the methodology and [docs/benchmarks.md#results](docs/benchmarks.md#results) for the full results table, per-design latency, per-design coverage, limitations, and caveats.

## Documentation

| page | contents |
|---|---|
| [docs/how-it-works.md](docs/how-it-works.md) | request flow, shadow vs. serve, question lifecycle, novelty forwarding, learning queue, namespaces |
| [docs/diagrams.md](docs/diagrams.md) | further diagrams: request lifecycles, cost at scale, coverage by question, design race, ladder, question fit |
| [docs/graduation.md](docs/graduation.md) | out-of-fold CV, Wilson bounds, graduation modes, refits, audits, demotion, worked calculations |
| [docs/understudents/README.md](docs/understudents/README.md) | model design comparison table and selection criteria |
| [docs/understudents/lite.md](docs/understudents/lite.md) | input-embedding mean and max pooling, microsecond latency, topic and keyword signals |
| [docs/understudents/keywords.md](docs/understudents/keywords.md) | hashed character and word n-grams, no neural network |
| [docs/understudents/probe.md](docs/understudents/probe.md) | layer-8 token mean from the encoder, linear probe |
| [docs/understudents/deep.md](docs/understudents/deep.md) | per-token states, mean/max or learned attention pooling (default) |
| [docs/understudents/neighbors.md](docs/understudents/neighbors.md) | cosine nearest-neighbor vote over past lead answers |
| [docs/understudents/shape.md](docs/understudents/shape.md) | structural features: lengths, shares, markers, JSON types |
| [docs/understudents/blend.md](docs/understudents/blend.md) | n-grams, structural features, and input embeddings in one model |
| [docs/understudents/net.md](docs/understudents/net.md) | small neural network over encoder and structural features |
| [docs/understudents/checklist.md](docs/understudents/checklist.md) | greedily learned decision list of one-feature rules, abstains when none fires |
| [docs/understudents/memo.md](docs/understudents/memo.md) | exact and near-duplicate retrieval of past answers |
| [docs/understudents/rules.md](docs/understudents/rules.md) | LLM-generated regex rules validated on held-out examples |
| [docs/understudents/ladder.md](docs/understudents/ladder.md) | tiered evaluation ordered from cheapest to most expensive; the cheapest eligible tier answers |
| [docs/understudents/auto.md](docs/understudents/auto.md) | per-question candidate racing and lead-only fallback |
| [docs/leads/README.md](docs/leads/README.md) | per-lead setup pages: TypeSafe Jev, Fireworks Rune, Cloudflare Clef, OpenAI Decisions API |
| [docs/security.md](docs/security.md) | data Understudy sees and writes to disk, listener address, key handling |
| [docs/install.md](docs/install.md) | install channels, checksum verification, container port and volume |
| [docs/api.md](docs/api.md) | request and response schema, error passthrough, modes, flags, status endpoints, capture log, limits |
| [docs/examples.md](docs/examples.md) | graduating a gate, lead-only fallback on broad questions, replay benchmarking, train ladder |
| [docs/benchmarks.md](docs/benchmarks.md) | benchmark method, metrics, cost at scale, per-question results, limitations |
| [SPEC.md](SPEC.md) | wire types, statistical formulas, file formats, CLI flags, snapshot format |
| [AGENTS.md](AGENTS.md) | repository conventions for coding agents |
| [CONTRIBUTING.md](CONTRIBUTING.md) | building, testing, and sending a pull request |
| [SECURITY.md](SECURITY.md) | reporting a vulnerability privately |
| [TODO.md](TODO.md) | pending benchmarks, planned model designs, preview-release checklist, post-release work |
| [bench/](bench/) | `understudy-bench`, the Go harness for dataset synthesis, lead collection, replay, and reporting |

## License

Understudy is released under the MIT license (see [LICENSE](LICENSE)). `vendor/eg2cpu` is MIT OR Apache-2.0 (see its license files and `PATCHED.md`).
