# API

`understudy serve` is a `POST /v1/systemone` server (alias `POST /systemone`, plus `GET /healthz` and `GET /status`). A
System One caller switches to it by changing only its base URL. It speaks HTTP/1.1 and is meant for a local client on loopback
(no TLS). Any other path gets a 404; nothing is forwarded.

- [Starting it](#running)
- [The embedding model](#embedding-model)
- [Leads](#leads)
- [Request and response](#request-and-response)
- [Question types](#question-types)
- [Several questions in one request](#multiple-questions-per-request)
- [Local versus lead responses](#local-versus-lead-responses)
- [Errors are passed through](#error-passthrough)
- [Modes and serve options](#modes-and-serve-options)
- [Health and status](#health-and-status)
- [The capture log](#capture-log)
- [Limits](#limits)

How it decides is in [how it works](how-it-works.md); the statistics are in [graduation](graduation.md).

## Running

```bash
# 1. shadow (the default): forward, learn, and log what the understudent would have said
understudy serve --lead-url https://your-lead.example.com/v1/systemone

# 2. serve: answer locally once a question has graduated (what shadow learned is reloaded)
understudy serve --lead-url https://your-lead.example.com/v1/systemone --mode serve

# 3. race: call the lead at once, answer locally if the local answer is ready (and eligible) first
understudy serve --lead-url https://your-lead.example.com/v1/systemone --mode race

# forward only, with a capture log (nothing is learned)
understudy serve --lead-url https://your-lead.example.com/v1/systemone --mode forward
```

`--lead-url URL` is a catch-all route: every request goes to that lead whatever its `model`, and no config file is needed.
Otherwise requests are routed by their `model`, to the leads of your [lead config file](#config-file) (a lead's `route` is the
model pattern it answers, else its `model`), which are tried before `--lead-url`. Add routes with `--route MODEL=URL` (a System One endpoint) or
`--chat-route MODEL=URL` (an OpenAI-compatible chat endpoint) or `--clef-route MODEL` (a [Cloudflare Clef](leads/cloudflare-clef.md) route; the URL is built from `$CLOUDFLARE_ACCOUNT_ID`); the model is exact, or a prefix ending in `*`. Ignore the config
with `--no-default-routes`, and send a bearer key for a route from the environment with `--route-key-env ROUTE=ENV_VAR` (the
`--lead-url` route is named `lead`). Without a configured key, the client's own `Authorization` header is forwarded to the lead.
With no config, no `--route` and no `--lead-url`, `serve` does not start: no endpoint is built in.

Point callers at the proxy by base URL only:

```bash
# a client that appends /systemone to its base URL
export BASE_URL=http://127.0.0.1:8787/v1
# a client that takes a complete endpoint
export ENDPOINT=http://127.0.0.1:8787/v1/systemone
```

## Embedding model

`keywords`, `shape`, `memo` and `checklist` start with no model. The others load EmbeddingGemma 2 on first use, and fail
with `no EmbeddingGemma 2 model found ... run `understudy model fetch`` if it is missing. `understudy serve` defaults to
`--understudent auto`, which never fails for a missing model: it races the model-free designs (keywords, shape, checklist, memo;
and rules when a rules LLM is configured) while it downloads the model in the background (checksum-verified, one log line when
it starts and one when it finishes, a warning and carry on if it fails), then adds the encoder designs to the race.
`--no-model-download` turns the download off. To fetch it yourself ahead of time:

```bash
understudy model fetch                 # ~1.5 GB into the Hugging Face cache, SHA-256 verified, no token needed
understudy model fetch --dir /opt/eg2  # or into a directory of your choice ...
export EG2_MODEL_DIR=/opt/eg2          # ... and point the loader at it
understudy model status                # resolved directory, revision, and a hash check of each file (a few seconds)
```

The loader prefers the pinned revision's snapshot in the cache, then the newest other snapshot (with a warning that
names it). `EG2_MODEL_DIR` overrides both and is not hash-checked, so loading stays fast. Behind a mirror, set
`HF_ENDPOINT`. See [SPEC.md](../SPEC.md#model-lookup-and-fetch).

## Leads

A **lead** is the decision model Understudy sits in front of. Understudy itself speaks System One everywhere (`judge`, replay,
`serve`); a lead is whatever can answer that.

| kind | what it is | status |
|---|---|---|
| `systemone` (default) | An endpoint taking `{model, questions, state}` and returning `{answers}`, such as Jev or Rune. Native: the request and response are used as they are. | works today; every benchmark used it |
| `chat` | An OpenAI-compatible `POST /chat/completions` endpoint, driven by the adapter below. | works today; tested against a fake endpoint and a live one; not benchmarked |
| `clef` | Cloudflare Workers AI Clef and Clef Flash: a System One body in, the reply unwrapped from Cloudflare's envelope. See [Cloudflare Clef](leads/cloudflare-clef.md). | works today; tested live on 2026-11-02 |
| replay | `--replay-lead stream.jsonl`: recorded answers by id, offline. | works today |

### Config file

Leads are named in a TOML file: `$UNDERSTUDY_LEADS`, else `~/.config/understudy/leads.toml` (`$XDG_CONFIG_HOME` is honored). No
endpoint is built into the binary or the bench. A missing default file means no leads; a missing `$UNDERSTUDY_LEADS` file is an
error. [`examples/leads.example.toml`](../examples/leads.example.toml):

```toml
[leads.rune]                              # --lead rune
kind = "systemone"                        # the default
url = "https://systemone.example.com/v1/systemone"
model = "example-router"
key_env = "SYSTEMONE_API_KEY"             # env var holding the bearer key; omit if none is needed
route = "example-*"                       # serve: the request models it answers (default: model, exact)

[leads.chat]
kind = "chat"
url = "https://api.fireworks.ai/inference/v1"     # a base URL, or the full .../chat/completions URL
model = "example-chat-model"
key_env = "FIREWORKS_API_KEY"
# also: reasoning_effort, json_schema (true), logprobs (true), confidence (0.9), max_tokens (2048)

[leads.clef]
kind = "clef"                                     # Cloudflare Workers AI: System One in, the envelope unwrapped
model = "clef-flash"
route = "clef*"                                   # serve: clef, clef-flash and @cf/cloudflare/<model>
# url = "https://api.cloudflare.com"              # the API base (default; or $CLOUDFLARE_API_BASE); optional for this kind
# account = "<account_id>"                        # default: $CLOUDFLARE_ACCOUNT_ID
# key_env = "CLOUDFLARE_API_TOKEN"                # the default for a clef lead

[rules]                                   # the chat LLM of the `rules` design (default: Fireworks' public API)
model = "example-chat-model"
```

```bash
understudy judge --snapshot snapshots/q1 --question q1.json --lead rune < states.jsonl         # a preset
understudy judge --snapshot snapshots/q1 --question q1.json --lead-kind chat \
  --lead-url https://api.fireworks.ai/inference/v1 --lead-model example-chat-model \
  --lead-key-env FIREWORKS_API_KEY < states.jsonl                                          # no config needed
```

An unknown preset is an error that says where presets come from. `judge` and `train` (the commands that call a lead) take
`--lead`, `--lead-url`, `--lead-model`, `--lead-kind` and `--lead-key-env`; `replay` answers from recorded probabilities and calls
no lead.

**Keys.** `--lead-key-env VAR` or a preset's `key_env` names an environment variable; its value is sent as
`Authorization: Bearer <value>`. The key is never taken from argv, never printed or logged, and never written to a snapshot (which
records the lead's `kind`, `url` and `model`) or a capture log. An unset variable is an error, so a missing key is not mistaken
for an open endpoint. An endpoint that injects credentials itself needs no `key_env`.

### Chat lead adapter

A decision request `{model, questions, state}` becomes **one chat call per question** (run concurrently), and each reply becomes
the System One answer for that question. One call per question, rather than one per request, because every call then has a single
small answer schema, a malformed answer costs one question and not the whole request, and the answer token's logprobs belong to
that question alone. The cost is that the state is sent once per question; a provider's prompt cache absorbs most of it.

The request, for a question and a state:

```json
{
  "model": "<the preset's model, else the request's model>",
  "messages": [
    {"role": "system", "content": "<SYSTEM>"},
    {"role": "user", "content": "<USER>"}
  ],
  "temperature": 0,
  "max_tokens": 2048,
  "reasoning_effort": "<only if configured>",
  "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": true,
    "schema": {"type": "object", "properties": {"answer": "<ANSWER SCHEMA>"}, "required": ["answer"], "additionalProperties": false}}},
  "logprobs": true,
  "top_logprobs": 5
}
```

`SYSTEM` is, verbatim:

```text
You are a precise decision function. You are given a question with its criteria and a state. Judge the state only against the question and its criteria. Reply with one JSON object of the form {"answer": ...} and nothing else: no prose and no code fence.
```

`USER` is this template. `{instructions}` is the question's instructions. `{criteria}` is one `- key: description` line per
criterion: for `noul` the `true` line and the `false` line; for `choice` each option in order; for `score` each level as `- 0: ...`,
`- 1: ...`, lowest first, under the heading `Criteria (levels, lowest first):` instead of `Criteria:`. `{format}` is the line for the
question's type. `{state}` is the state as it is when a string, else indented JSON.

```text
Question: {instructions}

Criteria:
{criteria}

Answer format: {format}

State:
{state}
```

| type | `{format}` | `<ANSWER SCHEMA>` | answer read as |
|---|---|---|---|
| `noul` | `{"answer": true} or {"answer": false} (a JSON boolean).` | `{"type": "boolean"}` | class 1 for true, 0 for false (`"yes"`, `"no"`, `1`, `0` are also read) |
| `choice` | `{"answer": "<option>"} where <option> is exactly one of: a, b, c.` | `{"type": "string", "enum": [options]}` | the option (exact, else case-insensitive) |
| `score` | `{"answer": <level>} where <level> is an integer from 0 to k-1.` | `{"type": "integer", "enum": [0, ..., k-1]}` | the level (a numeric string is also read) |

**Schema first, text second.** `response_format` and `logprobs` go first. If the endpoint answers 400 or 422, the call is repeated
once without them (and later calls start without them), and the JSON is parsed out of the text: the whole reply, a fenced block,
or the first balanced `{...}`. A reply with no readable answer, or one naming something outside the question's options, is an
error. Understudy never guesses.

**Probabilities.** System One returns a probability per class; a chat API does not. So:

- **Logprobs, when returned.** At the answer token, the alternatives that identify exactly one option by prefix (case-insensitive)
  give that option's probability mass. The masses are renormalized (floor 1e-4). They are used only when they hold at least half
  the probability and their argmax is the option the text named.
- **Otherwise a fixed confidence.** The chosen option gets **0.9** (`confidence` in the preset changes it, clamped to [0.55, 1]: 0.5 would tie a two-class question);
  the others share the rest equally. This is a documented stand-in, not a calibrated probability, and the response marks it:
  `"understudy": {"adapter": "chat", "probabilities": {"<question>": "logprobs" | "fixed"}}`. Learners fit these soft labels like
  any lead's answers, so they are coarser than a native lead's.

The answers are then built like a System One lead's: `{"type":"noul","noul":p_yes}`, `{"type":"choice","choice",...}`,
`{"type":"score","score":expected,...}`, with `usage` summed over the calls.

**In `serve`.** A route with `kind = "chat"` (or `--chat-route`) keeps the drop-in contract: callers still send System One and get
System One back, whether the request is forwarded, audited or raced. A forward is not byte-exact there (the body is the adapter's)
and carries `understudy.adapter`. A chat endpoint's error status and body are relayed unchanged (a 429 stays a 429); a reply with
no readable answer is a 502 `bad_gateway`. A chat endpoint's `Retry-After` is relayed too. At most 8 questions of one request run at once, and a chat response over `--max-response-mib` is a 502 `upstream_response_too_large`, as for a System One route.

## Request and response

The request is exactly what the lead takes:

```json
{
  "model": "jev-latest",
  "questions": {
    "urgent": {
      "type": "noul",
      "instructions": "Is this support ticket urgent?",
      "criteria": {
        "true": "the customer cannot work, loses money, or reports a security problem",
        "false": "a question, a request, or a complaint that can wait"
      }
    }
  },
  "state": {
    "subject": "Cannot log in since this morning",
    "body": "Our whole team is locked out and we have a launch today."
  }
}
```

```bash
curl -s -i http://127.0.0.1:8787/v1/systemone \
  -H 'content-type: application/json' \
  -d @request.json
```

The response, when the lead answers:

```http
HTTP/1.1 200 OK
content-type: application/json
x-understudy: lead

{"model":"jev-latest","answers":{"urgent":{"type":"noul","noul":0.97}},"usage":{"input_tokens":412,"cached_input_tokens":0,"output_tokens":9}}
```

The body is the lead's bytes, unchanged. The shapes below are as the leads return them.

A request body sent with `Content-Encoding: zstd` (the TypeSafe SDK does this for states of 1 KiB or more) is decoded to be read, and goes to the lead as it was sent.

## Question types

A question is `{"type", "instructions", "criteria"}`. There is no `bool` type: a yes/no decision is a `noul` question.

### noul (yes/no)

`criteria` is `{"true": "...", "false": "..."}`. The answer is `{"type":"noul","noul": p_yes}`, with `p_yes` in `[0, 1]`.

```json
{"type": "noul", "instructions": "Does this passage mention a refund?",
 "criteria": {"true": "the passage says a refund is offered or given", "false": "it does not"}}
```
```json
{"type": "noul", "noul": 0.9671}
```

### choice (one of several options)

`criteria` is an object `{option: description, ...}`. The answer names the chosen option, with probabilities and a
confidence (the highest probability).

```json
{"type": "choice", "instructions": "Which queue should this ticket go to?",
 "criteria": {"billing": "charges, refunds, invoices", "bug": "something is broken", "howto": "how do I do X"}}
```
```json
{"type": "choice", "choice": "billing", "confidence": 0.8123,
 "probabilities": {"billing": 0.8123, "bug": 0.1102, "howto": 0.0775}}
```

### score (ordered levels)

`criteria` is an **ordered array** of level descriptions. The answer gives the expected level, the probabilities per
level index and a confidence.

```json
{"type": "score", "instructions": "How polite is this reply?",
 "criteria": ["rude", "curt", "neutral", "warm"]}
```
```json
{"type": "score", "score": 2.31, "confidence": 0.58,
 "probabilities": {"0": 0.02, "1": 0.12, "2": 0.58, "3": 0.28}}
```

A lead answer that is a bare `{"choice": "billing"}` (no probabilities) teaches as the label `billing`. An answer that does not
parse teaches nothing.

## Multiple questions per request

Questions share one state. The answers keep the names the caller used:

```bash
curl -s http://127.0.0.1:8787/v1/systemone -H 'content-type: application/json' -d '{
  "model": "jev-latest",
  "questions": {
    "urgent": {"type": "noul", "instructions": "Is this support ticket urgent?",
               "criteria": {"true": "cannot work or loses money", "false": "can wait"}},
    "queue":  {"type": "choice", "instructions": "Which queue should this ticket go to?",
               "criteria": {"billing": "charges and refunds", "bug": "something is broken"}}
  },
  "state": {"subject": "Charged twice", "body": "My card was charged two times for May."}
}'
```

```json
{"model":"jev-latest",
 "answers":{"urgent":{"type":"noul","noul":0.12},
            "queue":{"type":"choice","choice":"billing","confidence":0.94,
                     "probabilities":{"billing":0.94,"bug":0.06}}},
 "usage":{"input_tokens":530,"cached_input_tokens":0,"output_tokens":21}}
```

- A learner is identified by the question **spec** (type, instructions, criteria), not its name. Identical specs under
  different names, even in one request, share one learner.
- In `serve` mode the request is answered locally only if **every** question may be. One question still learning
  forwards the whole request.
- A question the learners cannot parse (an unknown `type`) is opaque: forwarded as is, never learned, and any request
  containing one is forwarded whole.

## Local versus lead responses

Every response carries an `x-understudy` header:

| value | meaning |
|---|---|
| `lead` | forwarded to the lead; its response, byte for byte |
| `local` | answered by the understudent (also `/healthz` and `/status`) |
| `error` | the proxy's own error (bad request, unknown model, upstream unreachable) |

A local answer in `serve` mode:

```http
HTTP/1.1 200 OK
content-type: application/json
x-understudy: local

{"model":"jev-latest","answers":{"urgent":{"type":"noul","noul":0.0412}},"usage":{"input_tokens":0,"cached_input_tokens":0,"output_tokens":0}}
```

- `model` is the request's model, and `answers` keys equal the submitted names.
- `usage` is all zeros: nothing was spent upstream.
- Probabilities are rounded to four decimals and apportioned so they sum to exactly 1 at that precision; the
  choice and its confidence come from the printed probabilities.
- About one local answer in ten (`--audit-rate`) is also sent to the lead in the background to audit the understudent. Those
  calls cost the lead's price; nothing else does.

## Error passthrough

Upstream errors are never turned into 200s and never retried by the proxy:

| status | what happens |
|---|---|
| 400 (for example `max_tokens_exceeded`), 402 (billing), 413, 422 | the lead's status, body and headers, unchanged, `x-understudy: lead` |
| 429, 503 | unchanged, including `Retry-After` in either form (delta-seconds or an HTTP date) |
| other 5xx, a malformed JSON body | unchanged |
| 502, `upstream_unreachable` | a transport failure (connection refused, reset); `x-understudy: error`; callers should treat it as retryable |
| 504, `upstream_unreachable` | the upstream timed out (`--timeout-secs`, default 120) |
| 502, `upstream_response_too_large` | the upstream response was over `--max-response-mib` (default 256); never truncated |
| 400, `invalid_request` | the body is not a JSON object with a string `model` |
| 400, `unknown_model` | no route matches the model |
| 404, `not_found` | any other path that has no upstream to pass it to (more than one route, no `--lead-url`, and no routed `model` in the body) |
| 400/408/413/431/501/505, `bad_request` | the request's own HTTP framing was refused (see [limits](#limits)) |
| 503, `too_many_connections` | over `--max-conns`; carries `Retry-After: 1` |

The proxy's own errors have the shape:

```json
{"error": {"type": "unknown_model", "message": "no upstream is configured for model \"gpt-x\""}}
```

The upstream response bytes are relayed raw: no decompression, `Content-Encoding` kept, every value of a repeated header
kept, and the `Connection`-nominated headers dropped. The caller's end-to-end request headers (`Idempotency-Key`,
`Accept`, correlation ids) go upstream; only hop-by-hop ones, `Host` and `Content-Length` do not.

## Modes and serve options

| `--mode` | behavior |
|---|---|
| `forward` | pass-through plus the capture log; nothing is learned |
| `shadow` (default) | forward everything, learn, log what the understudent would answer |
| `serve` | answer locally when every question has graduated and is eligible; else forward as in shadow |
| `race` | call the lead immediately and compute the local answer beside it; the first usable result is returned (a local answer, or the lead's response), and the lead's response is never returned after a local answer, only audited |

`race` is the latency-first mode. A forwarded request is the lead's own latency (the lead call starts before any feature is
computed, so Understudy adds what shadow adds, about 0.3 ms), and a graduated, confident request is answered at local
speed while the lead call finishes in the background as an audit. The price is that **every** request reaches the lead,
so no lead calls are saved: race buys latency, and `serve` is the mode that saves lead calls (see
[Cost at scale](benchmarks.md#cost-at-scale)). A lead error after a local
answer went out is only logged; with no local answer, errors pass through unchanged.

### serve options

| flag | default | meaning |
|---|---|---|
| `--lead-url URL` | none | a catch-all System One lead for any model (the route is named `lead`); the caller's `Authorization` goes to it unless `--route-key-env lead=VAR` is set |
| `--route`, `--chat-route`, `--clef-route`, `--route-key-env`, `--no-default-routes` | none | more routes, and their keys ([above](#running)) |
| `--bind` | `127.0.0.1:8787` | listen address |
| `--mode` | `shadow` | `forward`, `shadow`, `serve` or `race` (above) |
| `--snapshot-dir DIR` | `understudy-snapshots` | persist each namespace's learners under `DIR/<upstream>__<model>-<hash>`; a restart reloads them in any mode |
| `--no-snapshots` | off | keep learning in memory only |
| `--understudent` | `auto` | the design; `auto` starts with the model-free ones and adds the encoder designs when the model is ready |
| `--no-model-download` | off | never fetch the embedding model in the background (`auto` then keeps to the model-free designs until `understudy model fetch`) |
| `--capture FILE` / `--no-capture` | `understudy-capture.jsonl` | the capture log |
| `--capture-bodies` | off | put full request and response bodies in the log (mind what states contain) |
| `--tiers`, `--candidates`, `--target`, `--minority`, `--audit-rate`, `--min-examples`, ... | as `understudy judge` | the learning configuration ([graduation](graduation.md)) |
| `--audit-rate` | 0.1 | share of local answers also audited by the lead |
| `--max-audits` | 4 | concurrent background audit calls (extra ones are skipped) |
| `--race-max-inflight` | 64 | `race`: most lead calls in flight at once; past it a request behaves as in `serve` for that request |
| `--prefer-fast EPS` | off | `--understudent auto`: serve the cheapest graduating candidate (by a deterministic cost model) whose coverage is within EPS of the best ([auto](understudents/auto.md#prefer-fast)) |
| `--local-wait-ms` | 5 | how long a request waits for a busy namespace before it is forwarded |
| `--queue` | 1024 | learning queue per namespace; when full jobs are dropped and counted |
| `--save-every` | 200 | save snapshots every N learned requests (also at shutdown and when idle) |
| `--timeout-secs` | 120 | upstream request timeout (for a passed-through stream, the longest wait for the next bytes) |
| `--max-conns` | 512 | concurrent connections (a thread each); more get a 503 |
| `--max-body-mib` | 64 | largest request body |
| `--max-response-mib` | 256 | largest upstream response relayed |
| `--max-namespaces` | 16 | namespaces (a worker and snapshot each); beyond it forward-only |
| `--max-learners` | 256 | learners per namespace; beyond it a new spec is forwarded and never learned |
| `--read-deadline-secs` | 60 | total time to receive one request |
| `--write-timeout-secs` | 30 | total time for the client to take a response |

A request header `x-understudy-group: G` makes the request's examples group `G` for cross-validation, so one game or
conversation is never split across folds.

## Health and status

```bash
curl -s http://127.0.0.1:8787/healthz
```
```
ok
```

```bash
curl -s http://127.0.0.1:8787/status
```
```json
{
  "mode": "serve",
  "requests": 1840, "local": 612, "lead": 1228,
  "upstream_errors": 0, "rejected": 1,
  "learn_queued": 1228, "learn_dropped": 0, "learned": 1228, "learn_errors": 0,
  "audits": 58, "audits_skipped": 0, "local_busy": 2,
  "namespaces_refused": 0, "learners_refused": 0, "post_dropped": 0, "response_too_large": 0,
  "namespaces": [
    {
      "namespace": "jev__jev-latest-3fa1c2",
      "route": "jev", "model": "jev-latest",
      "error": null, "pending": 0, "fit_version": 14,
      "learners": [
        {"spec_hash": "c909de5a3f108856", "state": "graduated", "examples": 421, "type": "noul"},
        {"spec_hash": "5be1a0e47d92c310", "state": "learning",  "examples": 87,  "type": "choice"}
      ]
    }
  ]
}
```

A learner's `state` is `learning`, `graduated`, `demoted` or `lead_only`. A namespace whose saved snapshot belongs to a
different upstream shows a clear `error` and forwards everything. `learners` is the string `"busy"` while a refit holds the
namespace. `understudy status --snapshot DIR` prints each learner's mode, threshold, readout, cross-validated agreement
and coverage, and audit window from a saved snapshot, without loading the model.

## Capture log

`--capture` writes JSONL, one line per request, plus `kind: "audit"` lines for background audits. A shadow-mode line:

```json
{"ts":"2026-11-02T14:03:21.482Z","kind":"request","model":"jev-latest","upstream":"jev","mode":"shadow",
 "request_bytes":612,"routed":"lead","latency_ms":418.2,"upstream_ms":417.9,"status":200,
 "usage":{"input_tokens":412,"cached_input_tokens":0,"output_tokens":9},
 "state_sha":"9d0c...","learning":"learned","refits":[],
 "questions":[{"name":"urgent","spec_hash":"c909de5a3f108856","type":"noul","opaque":false,
   "lead":"true",
   "understudent":{"label":"true","probs":{"false":0.04,"true":0.96},"eligible":true,"novel":false,
                 "demoted":false,"would_answer_local":true,"mode":"symmetric","readout":"keywords"},
   "agree":true,"learner_state":"graduated","examples":421}]}
```

| field | meaning |
|---|---|
| `ts`, `kind`, `model`, `upstream`, `mode` | when, `request` or `audit`, which route and mode |
| `routed` | `local` or `lead` |
| `race_id`, `local_ms`, `lead_ms` | `race` mode: a request id shared with its audit line, the local computation's time (null if the lead answered first) and the lead call's time (null on a local line) |
| `status` | the status the caller received |
| `latency_ms` | time to the response; `upstream_ms` is the upstream's share (null for local) |
| `usage` | the upstream's usage, zeros for local |
| `request_bytes`, `state_sha` | request size and a hash of the state |
| `learning` | `learned`, `dropped`, or why not (for example `not learned: the lead did not answer 2xx`, `forward-only: --max-namespaces reached`) |
| `refits` | any refits this request triggered: question, seconds, `n`, `mode`, `cv_agreement`, `cv_coverage` |
| `questions[]` | per question: `name`, `spec_hash`, `type`, `opaque`, `lead` (the lead's option), `understudent` (its `label`, `probs`, `eligible`, `novel`, `demoted`, `would_answer_local`, `mode`, `readout`), `agree`, `learner_state`, `examples` |
| `kind: "audit"`, `race_id` | `race` mode, after a local answer: the lead's late response, with `lead_ms` and per question `served`, `lead` and `agree` (the audit's agreement) |
| `served` | on audits: the label that was served, the `fit_version`, and whether the audit was counted (`stale` across a refit) |
| `group` | present when `x-understudy-group` was sent |
| `request`, `response` | the parsed bodies, only with `--capture-bodies` |

## Limits

- **Namespaces and learners.** A namespace (route + model + upstream URL + route-config hash) exists only after a 2xx upstream
  answer with an `answers` object for that model. `--max-namespaces` and `--max-learners` cap them; beyond a cap a request is
  forward-only and counted (`namespaces_refused`, `learners_refused`).
- **Queues.** The learning queue is bounded per namespace (`--queue`) and drops when full (`learn_dropped`); post-response work
  that finds its queue full is dropped and counted (`post_dropped`), never run on the connection thread. Dropping loses a
  training example, never a response.
- **HTTP framing is strict.** Duplicate or conflicting `Content-Length`, `Transfer-Encoding` together with `Content-Length`, any
  coding but `chunked`, bare line feeds and a chunk not followed by CRLF are errors that close the connection. More than 32
  trailers are refused. A request has `--read-deadline-secs` in total to arrive and a response `--write-timeout-secs` to be
  taken. 204 carries no `Content-Length`; 304 keeps the upstream's.
- **A busy namespace forwards.** In `serve` mode, if a refit holds the namespace longer than `--local-wait-ms`, the request is
  forwarded (`local_busy`).
- **Never point a live production caller at it** until it has run in shadow mode and its capture log agrees with the lead.
