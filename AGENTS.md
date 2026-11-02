# AGENTS.md

Guidance for coding agents working in the Understudy repo. Read [README.md](README.md) first, then [docs/](docs/) for how
it works ([how-it-works](docs/how-it-works.md), [graduation](docs/graduation.md),
[understudents](docs/understudents/README.md), [api](docs/api.md), [examples](docs/examples.md),
[benchmarks](docs/benchmarks.md), [security](docs/security.md)), and [SPEC.md](SPEC.md) for the precise reference (statistics, wire types, files, CLI,
snapshot format).

## Docs

- Docs describe the **current state** of the code, not its history. Update the page and SPEC.md in the same change as the code.
- Use generic examples only ("is this support ticket urgent?", "does this passage mention a refund?", "is this tool output
  an error?"). No employer, company, or work-project names, and no private dataset text.
- Frame cost at scale. Every call answered locally is a lead call never paid for, and savings grow linearly with call volume
  while the understudent's own cost is CPU time. Don't describe per-call savings as small or negligible; show them at volume
  (e.g. per million or billion calls, at the measured and at larger-model rates) next to the CPU cost. Treat cost, latency
  and availability as co-equal wins.
- Writing standard: no-nonsense, technical, and straightforward, like a senior engineer documenting a tool for other engineers.
  Headings are plain nouns naming the topic ("Graduation", "API", "Benchmarks"), never a description of the section's
  length or style ("in brief", "at a glance", "the short version").
- Register: README and docs are neutral third person. No "I", "we", "lol", "kinda", or other casual asides. Kyle's
  launch post is a source of ideas, not of register: tweets can be first person, the repo is not.
- Always use the Oxford comma ("a, b, and c"), in docs, code comments, commit messages, and status notes.
- No meta-commentary or hype: cut "it's worth noting", "in short", "simply", "just", "powerful", "seamless", "earns its keep",
  rhetorical questions, exclamation marks, emoji, and em-dash asides.
- State facts and numbers directly, in short declarative sentences. Use a table or code block when it carries the information
  better than prose. Keep every limitation and caveat.
- American spelling (license, neighbor, normalized, labeled, behavior).
- Every Mermaid block must render on GitHub (flowchart, sequenceDiagram, stateDiagram-v2, xychart-beta). Check relative links.

## Data boundaries

- data-private/ is gitignored and holds nothing that ships. Keep third-party datasets out of git unless their license allows redistribution; benchmarks report only the synthetic sets in `data/`.
- Committed runs/ are history: leave existing run outputs unchanged, and keep run artifacts text-free.

## Work conventions

- Commit as the repository's human maintainer, with no AI attribution of any kind.
- Keep `cargo test --release` and `(cd bench && go vet ./... && go test ./...)` green.
- Never point a live production caller at `understudy serve`. Use shadow or replay only, unless the maintainer says otherwise.
