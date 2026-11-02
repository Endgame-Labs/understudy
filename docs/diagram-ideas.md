# Diagram ideas

Topic diagrams for possible use in the docs, one idea each. Every block renders on GitHub (checked with mermaid-cli 11.4.2).

## 1. Question lifecycle

Emphasizes: a question earns local answers statistically and loses them when audits disagree.

```mermaid
stateDiagram-v2
    [*] --> learning: first call
    learning --> graduated: Wilson low ≥ target<br/>on out-of-fold CV
    learning --> lead_only: nothing learnable (auto)
    graduated --> demoted: audits disagree
    demoted --> learning: keep learning
    graduated --> graduated: 1 in 10 audited
```

## 2. One request, shadow versus serve

Emphasizes: shadow is risk-free, and serve is the same pipeline with the local branch switched on.

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
        Note over U: logs what it would have said
    else serve, question graduated
        U-->>C: local answer (≈0.1 ms)
        U--)L: audit 1 in 10
    end
```

## 3. Cost at scale

Emphasizes: every local answer is a lead call never paid for, while the CPU cost is a rounding error.

```mermaid
flowchart TD
    P["1 billion decision calls"] --> S{"understudy"}
    S -- "72% answered locally" --> Z["720M lead calls never made<br/>$45k saved at $0.042/M tokens<br/>up to $3.2M at larger-model prices"]
    S -- "28% forwarded" --> L["lead"]
    Z --> CPU["CPU cost of those answers:<br/>≈ $2 (cheap designs)"]
```

## 4. Where it helps: measured coverage

Emphasizes: real benchmark numbers per question type (share answered locally at 91–98% agreement with the lead).

```mermaid
xychart-beta
    title "Calls answered locally (%), Jev lead"
    x-axis ["preference", "yes/no gate", "tool output", "retry?", "errors", "passage", "lookup"]
    y-axis "coverage %" 0 --> 100
    bar [80, 69, 58, 53, 25, 14, 0]
```

## 5. The design race, per question

Emphasizes: `auto` races several understudents on each question and serves the cheapest safe winner.

```mermaid
flowchart TD
    Q["one question's lead answers"] --> K["keywords"] & LI["lite"] & PR["probe"] & D["deep"]
    K & LI & PR & D --> CV["grouped 5-fold CV<br/>Wilson low ≥ 0.90?"]
    CV -- "yes" --> W["serve the best coverage<br/>(cheaper wins a tie)"]
    CV -- "never, after 600 examples" --> LO["lead-only"]
```

## 6. The ladder: cheapest tier first

Emphasizes: each call tries the cheapest design that may answer, and only falls through to the lead at the end.

```mermaid
flowchart LR
    C["call"] --> K["keywords<br/>≈0.1 ms"]
    K -- "can't answer" --> LI["lite"]
    LI -- "can't answer" --> PR["probe"]
    PR -- "can't answer" --> D["deep<br/>≈55 ms"]
    D -- "can't answer" --> L["lead<br/>≈100 ms"]
    K & LI & PR & D -- "graduated, confident" --> A["answer"]
```

## 7. Before and after

Emphasizes: drop-in, because only the base URL changes.

```mermaid
flowchart LR
    subgraph before
        A1["your pipeline"] -- "every call" --> L1["lead"]
    end
    subgraph after
        A2["your pipeline"] -- "same API" --> U["understudy"]
        U -- "learned questions" --> R["local answer"]
        U -- "the rest" --> L2["lead"]
    end
```

## 8. Race mode: never slower than the lead

Emphasizes: the latency-first mode, where the lead call starts at once and a ready local answer wins.

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

## 9. Question fit

Emphasizes: a guide to where it helps and where it stays lead-only.

```mermaid
flowchart TD
    Q["a decision your code asks a model"] --> R{"asked often,<br/>about similar inputs?"}
    R -- "no" --> LO["lead-only<br/>(costs ≈0.03 ms overhead)"]
    R -- "yes" --> V{"is the answer visible<br/>in the input text?"}
    V -- "no: deep reasoning,<br/>best-of-N judging" --> LO
    V -- "yes" --> B{"both answers common?"}
    B -- "no: one answer 99%" --> LO
    B -- "yes" --> H["good fit: judging, routing,<br/>classification, yes/no gates"]
```

## 10. The learning loop

Emphasizes: the flywheel, where lead answers become training data and audits check agreement.

```mermaid
flowchart LR
    L["lead answers"] --> E["examples<br/>(last 2,000 per question)"]
    E --> F["refit every 100<br/>out-of-fold CV"]
    F --> G{"graduate?"}
    G -- "yes" --> A["local answers"]
    G -- "not yet" --> L
    A --> AU["audits (1 in 10)"]
    AU -- "agree" --> A
    AU -- "drift" --> F
```
