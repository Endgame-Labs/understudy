# AGENTS.md

Guidance for coding agents working in the Understudy repo. Read [README.md](README.md) first, then [docs/](docs/) for how
it works ([how-it-works](docs/how-it-works.md), [graduation](docs/graduation.md),
[understudents](docs/understudents/README.md), [api](docs/api.md), [examples](docs/examples.md),
[benchmarks](docs/benchmarks.md), and [security](docs/security.md)), and [SPEC.md](SPEC.md) for the precise reference
(statistics, wire types, files, CLI, and snapshot format). Contribution steps are in [CONTRIBUTING.md](CONTRIBUTING.md);
vulnerability reports follow [SECURITY.md](SECURITY.md).

## Build and test

```bash
cargo build --release                                   # target/release/understudy
cargo test --release                                    # Rust tests
(cd bench && go vet ./... && go test ./...)             # Go benchmark harness
```

Keep both test suites green.

## Layout

| path | contents |
|---|---|
| `src/` | the `understudy` binary and library: the serve proxy, leads, understudent designs, graduation, and snapshots |
| `tests/` | Rust integration tests |
| `bench/` | `understudy-bench`, the Go harness that synthesizes datasets, replays lead answers, and writes reports |
| `vendor/eg2cpu/` | the vendored EmbeddingGemma 2 CPU engine (MIT OR Apache-2.0), patched for per-token states |
| `docs/` | how it works, graduation, designs, API, leads, benchmarks, security, and diagrams |
| `data/` | synthetic benchmark datasets and recorded lead answers (see [data/README.md](data/README.md)) |
| `runs/` | committed benchmark runs over `data/` |

## Docs

- Docs describe the **current state** of the code, not its history. Update the page and SPEC.md in the same change as the code.
- Use generic examples only ("is this support ticket urgent?", "does this passage mention a refund?", "is this tool output
  an error?"). No customer names, private data, or internal hostnames.
- Report cost per million or billion calls, at measured and larger-model rates, next to the CPU cost; keep latency and
  availability alongside.
- Writing standard: no-nonsense, technical, and straightforward, like a senior engineer documenting a tool for other engineers.
  Headings are plain nouns naming the topic ("Graduation", "API", "Benchmarks"), never a description of the section's
  length or style ("in brief", "at a glance", "the short version").
- README and docs are neutral third person. No "I", "we", "lol", "kinda", or other casual asides.
- Always use the Oxford comma ("a, b, and c"), in docs, code comments, commit messages, and status notes.
- No meta-commentary or hype: cut "it's worth noting", "in short", "simply", "just", "powerful", "seamless", "earns its keep",
  rhetorical questions, exclamation marks, emoji, and em-dash asides.
- State facts and numbers directly, in short declarative sentences. Use a table or code block when it carries the information
  better than prose. Keep every limitation and caveat.
- American spelling (license, neighbor, normalized, labeled, behavior).
- Every Mermaid block must render on GitHub (flowchart, sequenceDiagram, stateDiagram-v2, xychart-beta). Check relative links.

## Data boundaries

- data-private/ is gitignored and holds nothing that ships. Keep third-party datasets out of git unless their license allows
  redistribution; benchmarks report only the synthetic sets in `data/`.
- Committed runs/ are history: leave existing run outputs unchanged, and keep run artifacts text-free.

## Work conventions

- Commit as the repository's human maintainer, with no AI attribution of any kind.
- Agents working in this repo test with shadow mode or replay, and never send traffic from production systems they don't own.
