# Vendored eg2cpu (patched)

Copied from [undrgrnd-labs/hunch](https://github.com/undrgrnd-labs/hunch) at commit `22cf730`, `eg2cpu/`
(MIT OR Apache-2.0; both licence files kept here).

Changes for Understudy:
- `Model::embed_batch_tokens` and `TokenStates`: per-token hidden states after chosen layers, which Understudy's
  readouts (max and attention pooling) need. Upstream only exposes the token means.
- `Model::embed_batch_tokens_partial` / `resume_batch_tokens` and `TokenPartial`: a per-token-state pass that stops at
  some layer and can be continued to deeper layers without recomputing (Understudy's `ladder` runs `probe`'s layers 0..8
  and, only if `probe` can't answer, continues the same pass to layer 12 for deep). Same layer code (`run_layers`) as
  `embed_batch_tokens`, so the states are identical.
- `Model::input_embeddings(ids)`: the scaled rows of the input embedding table (no layer runs), a public wrapper of
  the private `embed_ids`, for Understudy's `lite` understudy.
- `crate-type` is `rlib` only, the release profile moved to Understudy's `Cargo.toml`, and `src/bin/` is dropped.

Everything else is byte-for-byte upstream.
