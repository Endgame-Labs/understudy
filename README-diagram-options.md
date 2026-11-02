# README diagram options: the system overview

Ten ways to draw the same picture at the top of [README.md](README.md). Each shows the whole system at a glance:
- a caller and Understudy in front of a lead;
- local answers only after a question graduates;
- everything else forwarded to the lead and learned from;
- audits and demotion keeping it honest;
- the design race choosing a model per question.

Option 0 is the current diagram. The form changes from option to option, the subject doesn't. Every block renders on GitHub
(checked with mermaid-cli 11.4.2). Topic diagrams on single subjects are kept in [docs/diagram-ideas.md](docs/diagram-ideas.md).

## 0 (current)

```mermaid
flowchart LR
    C["caller"] --> U["understudy serve"]
    U -- "graduated, confident, not novel" --> A["local answer<br/>(CPU, microseconds to milliseconds)"]
    U -- "otherwise" --> L["lead: Jev or Rune"]
    L --> U
    U -. "learns off the hot path, audits, demotes" .-> U
```

## 1. Left to right, with the learner below

Emphasizes: the request path reads straight across, and learning sits under it.

```mermaid
flowchart LR
    C["caller"] --> U{"understudy:<br/>question graduated?"}
    U -- "yes" --> A["local answer<br/>(CPU)"]
    U -- "no" --> L["lead"]
    L --> R["lead's answer"]
    L -. "every answer" .-> T["learner: designs race<br/>per question"]
    T -. "graduates when safe" .-> U
    A -. "1 in 10 audited" .-> L
```

## 2. Top-down layers

Emphasizes: a stack of layers: the caller, Understudy, the learner, then the lead.

```mermaid
flowchart TD
    C["caller (unchanged code, new base URL)"] --> U["understudy serve"]
    U --> G{"graduated?"}
    G -- "yes" --> A["local answer"]
    G -- "no" --> L["lead"]
    L --> LR["learner<br/>race designs, refit, CV"]
    LR --> G
    A --> AU["audits → demote on drift"]
    AU --> G
```

## 3. Hub: Understudy between caller, lead and learner

Emphasizes: Understudy is a hub, and every arrow is one of its jobs.

```mermaid
flowchart TD
    U(("understudy"))
    C["caller"] <-->|"same API"| U
    U <-->|"forward + audits"| L["lead"]
    U <-->|"examples / graduations"| T["learner<br/>(design race per question)"]
    U -->|"graduated questions"| A["local answer"]
```

## 4. Swimlanes: caller, Understudy, lead

Emphasizes: who does what, in three lanes.

```mermaid
flowchart LR
    subgraph caller
        C["request"]
        R["answer"]
    end
    subgraph understudy
        G{"graduated?"}
        A["local model<br/>(won the design race)"]
        T["learn + audit"]
    end
    subgraph lead
        L["lead model"]
    end
    C --> G
    G -- "yes" --> A --> R
    G -- "no" --> L --> R
    L -.-> T -.-> G
```

## 5. Minimal: four boxes

Emphasizes: the shortest version that still tells the whole story.

```mermaid
flowchart LR
    C["caller"] --> U["understudy"]
    U -- "learned" --> A["local answer"]
    U -- "not yet" --> L["lead"]
    L -. "teaches + audits" .-> U
```

## 6. Hot path versus background

Emphasizes: the request path stays fast, and all the learning happens off it.

```mermaid
flowchart TD
    subgraph hot["hot path"]
        C["caller"] --> G{"graduated?"}
        G -- "yes" --> A["local answer"]
        G -- "no" --> L["lead"]
    end
    subgraph bg["background"]
        E["store example"] --> RC["design race:<br/>refit + CV"]
        RC --> GR["graduate or demote"]
        AU["audit sample"] --> GR
    end
    L -.-> E
    A -.-> AU
    GR -.-> G
```

## 7. Per question: several questions at different stages

Emphasizes: Understudy decides question by question, so one request stream holds questions at every stage.

```mermaid
flowchart LR
    C["caller"] --> U["understudy"]
    U --> Q1["is it urgent?<br/>graduated (deep)"] --> A["local answer"]
    U --> Q2["which team?<br/>learning"] --> L["lead"]
    U --> Q3["best move?<br/>lead-only"] --> L
    L -. "answers teach + audit" .-> U
```

## 8. CPU versus lead calls

Emphasizes: where the work and the money go: cheap CPU answers against paid lead calls.

```mermaid
flowchart LR
    C["caller"] --> U["understudy"]
    U -- "graduated questions" --> CPU["CPU answer<br/>≈0.1-55 ms, ≈free"]
    U -- "everything else" --> LC["lead call<br/>≈100 ms, paid per token"]
    LC -. "learn; race designs" .-> U
    CPU -. "audits (1 in 10)" .-> LC
```

## 9. Sequence diagram

Emphasizes: time order: the first calls go to the lead, and later calls are answered locally.

```mermaid
sequenceDiagram
    participant C as caller
    participant U as understudy
    participant L as lead
    C->>U: early calls
    U->>L: forward
    L-->>C: lead's answer (Understudy learns)
    Note over U: design race → question graduates
    C->>U: later calls
    U-->>C: local answer
    U--)L: audit 1 in 10 (demote on drift)
```

## 10. A loop around the lead

Emphasizes: the flywheel, where answers become training data and audits close the loop.

```mermaid
flowchart LR
    C["caller"] --> U{"understudy"}
    U -- "graduated" --> A["local answer"]
    U -- "otherwise" --> L["lead"]
    L --> T["learn: race designs<br/>per question"]
    T --> G["graduate"] --> U
    A --> AU["audit"] -- "drift" --> D["demote"] --> U
```
