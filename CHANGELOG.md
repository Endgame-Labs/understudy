# Changelog

All notable changes to Understudy are documented in this file. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/).

## [0.1.0-preview.2] - 2026-11-02

### Changed

- `understudy serve` defaults to `--mode serve`: questions are answered locally once they graduate, with no restart. Use `--mode shadow` to watch without answering.

### Fixed

- The shell installer command pins the release URL, because `releases/latest` returns 404 while only pre-releases exist.
- The `.deb` instructions use `curl` and the renamed asset (`understudy_0.1.0.preview.2_amd64.deb`) instead of `gh`.
- The Docker entrypoint passes `--version`, `--help`, and subcommands through to `understudy`, so `docker run IMAGE --version` works.
- The `understudy --help` about-text matches the package description.

## [0.1.0-preview.1] - 2026-11-02

First public preview release.

### Added

- `understudy serve`: an HTTP proxy that accepts System One requests (`POST /v1/systemone`), forwards them to a lead model,
  and learns from the lead's answers. Modes: `forward`, `shadow`, `serve`, and `race`. Local answers carry the
  `x-understudy: local` header. `GET /healthz` and `GET /status` report server health and per-question state.
- Question lifecycle: out-of-fold cross-validation, Wilson-bound graduation (with an opt-in sequential mode), novelty
  forwarding, sampled audits, and demotion back to the lead.
- 11 understudent designs: `lite`, `keywords`, `probe`, `deep`, `neighbors`, `shape`, `blend`, `memo`, `net`, `checklist`, and
  `rules`, plus the `ladder` and `auto` selectors that race or stack them per question.
- Lead adapters: System One (Jev and compatible endpoints), OpenAI-compatible chat endpoints (for example Fireworks Rune),
  and Cloudflare Clef, with lead presets read from a config file.
- Snapshots: per-question examples, fitted models, and audit windows saved to `--snapshot-dir`, so a restart keeps what was
  learned.
- Model fetch for the embedding model used by the encoder designs.
- A capture log (`--capture`) of requests, timings, and answers.
- The `understudy-bench` harness: replay and cold-start benchmarks over the synthetic datasets in `data/`, with committed
  results and cost-at-scale tables.
- Release pipeline: archives for x86_64 and aarch64 Linux and macOS with SHA-256 checksums, a shell installer, a Homebrew
  formula, `.deb` packages, and a container image.

[0.1.0-preview.2]: https://github.com/endgame-labs/understudy/releases/tag/v0.1.0-preview.2
[0.1.0-preview.1]: https://github.com/endgame-labs/understudy/releases/tag/v0.1.0-preview.1
