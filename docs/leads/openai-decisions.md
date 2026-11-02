# OpenAI Decisions API

**STATUS: planned.** The Decisions API is not System One: it needs a request and response translation adapter.

OpenAI's [guide](https://developers.openai.com/api/docs/guides/decisions.md) calls it generally available (the public beta
is over): `POST https://api.openai.com/v1/decisions`, model `gpt-6-luna` (the only one), `Authorization: Bearer`. Fireworks
documents an OpenAI-shaped `/inference/v1/decisions` too, so one adapter could serve both.

## What differs from System One

| | System One (Understudy) | OpenAI Decisions |
|---|---|---|
| state | `state` (string, object, or array) | `input` (string, or messages, `input_image`, `input_json` parts) |
| questions | object keyed by name | ordered array, each with `name` |
| yes/no | `type: "noul"`, `criteria: {true, false}` | `type: "predicate"` |
| choice | `criteria: {option: description}` | `choices: [{value, description}]` |
| score | `criteria: [..]` | `levels` |
| answers | object keyed by name, `noul` field | ordered array, `probability` field |
| probabilities | `{option: p}` | `[{value, probability}]` |
| usage | `input_tokens`, `cached_input_tokens`, `output_tokens` | adds `*_details`, `total_tokens` |

Answers can also be of type `refusal`. No `box`.

## Tested 2026-11-02

- A System One body posted to the Decisions endpoint (through a key-injecting proxy on the test machine), directly and
  through `understudy serve --route '*=...'`: 400 `Missing required parameter: 'input'`.
- A native Decisions body (one `predicate`, one `choice`): 200 with the array answers above.

## What is missing

An adapter that rewrites the request, calls `/v1/decisions`, and rewrites the answers back (with `noul` from `probability`,
probabilities as maps), as the `chat` kind does for chat completions. It is in [TODO](../../TODO.md).

## When it lands

Use a separate client for decision calls, with `base_url` set in code, for example
`OpenAI(base_url="http://127.0.0.1:8787/v1")` used only for decisions.

**Do not set `OPENAI_BASE_URL`: that would send chat, embeddings, images, and every other OpenAI call through Understudy too.**

```bash
understudy serve --lead-url https://api.openai.com/v1/decisions --route-key-env '*=OPENAI_API_KEY' --mode shadow
```
