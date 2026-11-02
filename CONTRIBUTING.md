# Contributing

## Build

Requirements: Rust 1.88 or newer, and Go 1.24 or newer for the benchmark harness.

```bash
cargo build --release                                   # target/release/understudy
(cd bench && go build -o ../target/understudy-bench .)  # the benchmark harness
```

The encoder designs need the EmbeddingGemma 2 checkpoint: `target/release/understudy model fetch` downloads and verifies
it (about 1.5 GB). The `keywords`, `shape`, `memo`, and `checklist` designs need no model.

## Test

```bash
cargo test --release
(cd bench && go vet ./... && go test ./...)
```

Both suites must pass before a change is merged. Replay benchmarks over the synthetic datasets in `data/` need no lead
calls; see [docs/examples.md](docs/examples.md).

## Pull requests

1. Fork the repository, and create a branch from `main`.
2. Keep each pull request to one change. Update the relevant page in `docs/` and [SPEC.md](SPEC.md) in the same change
   as the code.
3. Add or update tests for behavior changes.
4. Open the pull request against `main`, with a description of what changed and how it was tested.

Docs follow the writing standard in [AGENTS.md](AGENTS.md#docs): neutral third person, plain-noun headings, facts and
numbers, American spelling, and the Oxford comma. Datasets and run artifacts follow its
[data boundaries](AGENTS.md#data-boundaries).

## Security issues

Do not open a public issue for a vulnerability. Follow [SECURITY.md](SECURITY.md).

## License

Contributions are licensed under the [MIT License](LICENSE).
