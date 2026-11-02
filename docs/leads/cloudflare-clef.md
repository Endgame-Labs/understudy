# Cloudflare Clef and Clef Flash

**STATUS: works today** (`noul` and `choice` questions). Tested end to end on 2026-11-02 (see [Tested](#tested-2026-11-02)).

[Workers AI](https://developers.cloudflare.com/workers-ai/models/clef/index.md): `@cf/cloudflare/clef` (27B, 65,536-token
context) and `@cf/cloudflare/clef-flash` (9B, 24,576). Both take a System One request body as it is and answer System One, wrapped
in Cloudflare's API envelope:

```text
POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/cloudflare/clef-flash
Authorization: Bearer <Workers AI token>

{"model": "clef-flash", "state": {...}, "questions": {...}}
-> {"result": {"model": "clef-flash", "answers": {...}, "usage": {...}}, "success": true, "errors": [], "messages": []}
```

Questions are `noul`, `choice`, and `score` (at most 64, no `box`), plus an optional `images` field (a Clef extension). Cloudflare
does not call it System One compatible.

## Adapter

A `clef` lead (`src/clef.rs`) sends the caller's body unchanged and returns the envelope's `result` as the System One response
(status 200, `Content-Type: application/json`), so callers, learning, and audits see the same shape as from any System One lead.

- **Routing.** A request's `model` selects the route: `clef`, `clef-flash`, or `@cf/cloudflare/<model>`. The call goes to
  `{base}/client/v4/accounts/{account}/ai/run/@cf/cloudflare/{model}`. The base is `https://api.cloudflare.com`, or `url` in the
  config, or `$CLOUDFLARE_API_BASE`. The account is `account` in the config or `$CLOUDFLARE_ACCOUNT_ID`; it is never built in, and
  `serve` refuses to start without one. Cloudflare accepts only `clef` or `clef-flash` as the body's `model`, so a
  `@cf/cloudflare/<model>` name is rewritten to `<model>` in the body; any other body is sent byte for byte.
- **Failures.** `success: false`, or a non-empty `errors`, on a 2xx is a 502. A non-2xx keeps Cloudflare's status and
  `Retry-After`. The body is `{"error": {"type": "upstream_error", "message": "Cloudflare Workers AI: <messages>",
  "cloudflare_errors": [...]}}`. The response read is limited by `--max-response-mib`; a larger one is an error, never relayed truncated.
- **Auth.** A key from `key_env`, `--route-key-env` or, by default, `$CLOUDFLARE_API_TOKEN` is sent as `Authorization: Bearer`.
  With none configured and the variable unset, the caller's `Authorization` is forwarded. A key is never logged, put in argv, a
  snapshot or a capture log. A snapshot records the API base and the model, and its route URL shows `<account_id>` in place of the account.
- **Commands.** `understudy serve` (all modes, including audits), `judge`, and `train` (`--lead-kind clef`, `--lead-url` is the API base).

## Config

```toml
[leads.clef]
kind = "clef"
model = "clef-flash"        # the model `judge` and `train` call
route = "clef*"             # serve: the request models it answers (default: model, else clef*)
# url = "https://api.cloudflare.com"   # the API base; the default
# account = "<account_id>"             # default: $CLOUDFLARE_ACCOUNT_ID
# key_env = "CLOUDFLARE_API_TOKEN"     # the default for a clef lead; omit to forward the caller's Authorization
```

```bash
export CLOUDFLARE_ACCOUNT_ID=<account_id>
export CLOUDFLARE_API_TOKEN=<Workers AI token>
understudy serve --clef-route 'clef*' --mode shadow --snapshot-dir snapshots/
# or, with the lead config above:
understudy serve --mode shadow --snapshot-dir snapshots/
understudy judge --snapshot snapshots/q1 --question q1.json --lead-kind clef --lead-model clef-flash < states.jsonl
```

The caller sends a normal System One request to Understudy with `"model": "clef-flash"`; Understudy calls the public
`api.cloudflare.com` and answers as in [serve](../api.md#serve-options). Start in `shadow`, and move to `serve` once questions have graduated.

**Never change a global Cloudflare base URL to point at Understudy: every Workers AI call (chat, embeddings, images) would go through it, and it answers none of them.
Point only the decisions client at Understudy, with its URL set in code.**

## Tested 2026-11-02

Through a key-injecting proxy for the Cloudflare API, with `understudy serve --mode shadow --clef-route 'clef*'`:

- 20 requests each to `clef-flash` and `clef`, each with one `noul` and one `choice` question: 40 of 40 returned 200 with
  `x-understudy: lead` and a plain System One body (`model`, `answers`, `usage`; no `result` or `success`). A `noul` was 0.95 to 0.99 for
  urgent tickets and 0.01 to 0.07 for routine ones; `choice` answers carried `probabilities` and `confidence`.
- Graduation: 600 items of the synthetic `tool_failed` set (`data/tool_failed`) were replayed as live requests through `serve --mode shadow`
  in front of `clef-flash`. All 600 returned 200. The lead agreed with the dataset's label on 562 (93.7%). The learner for
  the question was `graduated` after the 600th: `symmetric` mode, cross-validated agreement 0.918 against the 0.9 target, coverage 0.873
  (0.66 at 300 examples, 0.69 at 400, 0.77 at 500).
- Cost: Clef Flash is about 5 neurons for a short call, so the 600 calls are about 3,000 neurons, about $0.03 at $0.011 per 1,000 neurons.
  Responses report tokens, not neurons, so this is an estimate. After graduation, every call answered locally is one never paid for.
- Errors: Cloudflare's own 400 (a `model` outside `clef` and `clef-flash`) and 422 came back with its status and message.

Not tested: `score`, `images`, `serve` mode against the live API (it is covered against a fake endpoint), `race`.

## Long state

Cloudflare's docs say a long state is truncated to fit the context. Observed: it was not. `usage.input_tokens` counted the whole
state, and a fact placed at the end of the state was answered correctly at every size up to the limit
(`choice` probability of the right code at least 0.98).

| model | state | `input_tokens` | result |
|---|---|---|---|
| `clef-flash` | about 2,000 and 3,500 words, fact at start, middle, end | 2,240 to 3,734 | correct |
| `clef-flash` | 12,000 and 18,000 words, fact at start, middle, end | 12,239 to 18,244 | correct |
| `clef-flash` | 24,000 words, fact at the end | 24,236 | correct |
| `clef-flash` | 26,000 words | | 422 |
| `clef-flash` | 30,000 words | | 413 |
| `clef` | 2,000, 12,000, 18,000, and 63,000 words, fact at the end | 2,240 to 63,242 | correct |
| `clef` | 68,000 words | | 422 |

A state over the context is an error on the day of the test, not a silent loss, and Understudy relays that status and message. The
reported 2,000-token limit was not seen. Cloudflare may change this; the check costs a few calls.
