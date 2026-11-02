# hunch_runner: optional external baseline

Replays a benchmark stream through [hunch](https://github.com/undrgrnd-labs/hunch) (Apache-2.0), another reflex-layer
proxy, so its results can sit next to understudent's. It is not a default system: ask for it with
`understudy-bench run --systems hunch,understudent:deep,...`.

`uv run` fetches `hunch` and `eg2cpu` from that public repository at the commit pinned in `pyproject.toml`
(`uv.lock` records the hashes), so the first run needs network access. Nothing from hunch is vendored here.
