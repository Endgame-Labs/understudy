# Auxiliary system diagrams

Further diagrams of how Understudy works, beyond the three in the [README](../README.md).

## Graduation lifecycle

A question earns local answers statistically and loses them when audits disagree. See [graduation](graduation.md).

```mermaid
stateDiagram-v2
    [*] --> learning: first call
    learning --> graduated: Wilson low ≥ target<br/>on out-of-fold CV
    learning --> lead_only: nothing learnable (auto)
    graduated --> demoted: audits disagree
    demoted --> learning: keep learning
    graduated --> graduated: 1 in 10 audited
```

## Request lifecycle: shadow vs serve

Shadow forwards every request and logs what the understudent would have said. Serve is the same pipeline with the local
branch enabled for graduated questions.

```mermaid
sequenceDiagram
    participant C as caller
    participant U as understudy
    participant L as lead
    C->>U: decision request
    alt shadow (default)
        U->>L: forward
        L-->>U: answer
        U-->>C: lead's answer
        Note over U: logs what the understudent would have said
    else serve, question graduated
        U-->>C: local answer
        U--)L: audit 1 in 10
    end
```

## Request lifecycle: race

Race starts the lead call immediately. A local answer that is graduated, confident, and ready first is returned, and the
lead's response is then used only as an audit.

```mermaid
sequenceDiagram
    participant C as caller
    participant U as understudy
    participant L as lead
    C->>U: request
    U->>L: forward immediately
    U-->>C: local answer, if graduated and confident
    L-->>U: lead's answer (an audit if local won)
    Note over C,U: not graduated: the caller gets the lead's answer
```

## Cost at scale

At 75% net local share and the measured lead rate of $0.042 per million input tokens, one billion calls leave 750 million
lead calls unmade. The CPU cost of a cheap design is about $2 per billion calls. Figures are from
[Cost at scale](benchmarks.md#cost-at-scale).

```mermaid
flowchart TD
    P["1 billion decision calls"] --> S{"understudy"}
    S -- "75% net answered locally" --> Z["750M lead calls never made<br/>$46,694 saved at $0.042 per M tokens<br/>$3,335,294 at $3.00 per M tokens"]
    S -- "25% forwarded" --> L["lead"]
    Z --> CPU["CPU cost of the understudent on all calls:<br/>$2.22 (cheap designs), $833 (encoder designs)"]
```

## Coverage by question

Share of calls answered locally over the cold-start stream, Jev as the lead, from
[Results](benchmarks.md#results). Questions that never graduate (`capital_city`, `code_lookup`, `customer_frustration`,
and `secret_needle`) score 0.

```mermaid
xychart-beta
    title "Calls answered locally (%), Jev lead, cold start"
    x-axis ["politer", "payment", "output", "retry", "tool", "passage", "incident", "capital"]
    y-axis "coverage %" 0 --> 100
    bar [80, 69, 58, 53, 25, 14, 11, 0]
```

The x-axis labels abbreviate `politer_reply`, `payment_flip`, `output_kind`, `retry_exception`, `tool_failed`,
`passage_answers`, `incident_severity`, and `capital_city`.

## Design race

`auto` fits several understudent designs on each question's examples and serves the best one that clears the bar. See
[auto](understudents/auto.md).

```mermaid
flowchart TD
    Q["one question's lead answers"] --> K["keywords"] & LI["lite"] & PR["probe"] & D["deep"]
    K & LI & PR & D --> CV["5-fold CV<br/>Wilson low ≥ 0.90?"]
    CV -- "yes" --> W["serve the best coverage<br/>(cheaper wins a tie)"]
    CV -- "none after 600 examples" --> LO["lead-only"]
```

## The ladder

Each call tries the cheapest tier that may answer and reaches the lead only when no tier does. Latencies are p50 from
[Latency](benchmarks.md#latency). See [ladder](understudents/ladder.md).

```mermaid
flowchart LR
    C["call"] --> K["keywords<br/>≈0.1 ms"]
    K -- "can't answer" --> LI["lite"]
    LI -- "can't answer" --> PR["probe"]
    PR -- "can't answer" --> D["deep<br/>66-89 ms"]
    D -- "can't answer" --> L["lead<br/>78-82 ms"]
    K & LI & PR & D -- "graduated, confident" --> A["answer"]
```

## Before and after

Adopting Understudy changes only the base URL the pipeline calls.

```mermaid
flowchart LR
    subgraph before
        A1["pipeline"] -- "every call" --> L1["lead"]
    end
    subgraph after
        A2["pipeline"] -- "same API" --> U["understudy"]
        U -- "graduated questions" --> R["local answer"]
        U -- "the rest" --> L2["lead"]
    end
```

## Question fit

Understudy helps on questions asked often, about similar inputs, with an answer visible in the text. Other questions stay
on the lead. See [Scope](benchmarks.md#scope).

```mermaid
flowchart TD
    Q["a decision your code asks a model"] --> R{"asked often,<br/>about similar inputs?"}
    R -- "no" --> LO["lead-only<br/>(forwarding overhead only)"]
    R -- "yes" --> V{"is the answer visible<br/>in the input text?"}
    V -- "no: lookups, needles in long text,<br/>comparing two changing texts" --> LO
    V -- "yes" --> B{"both answers common?"}
    B -- "no: one answer ~99%" --> LO
    B -- "yes" --> H["good fit: judging, routing,<br/>classification, yes/no gates"]
```
