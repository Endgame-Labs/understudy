# Fireworks Rune

**STATUS: works today** (text questions only).

Rune is a System One lead on Fireworks at `POST /inference/v1/systemone`. Understudy speaks it as is.

## In the caller

Make a client used only for decision calls, with its URL set in code, and leave the global Fireworks base URL alone:

```python
import requests, os

def decide(body):  # decision calls only
    return requests.post("http://127.0.0.1:8787/v1/systemone", json=body,
                         headers={"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}"}).json()
```

with `"model": "accounts/fireworks/models/rune"`.

**Do not change a global Fireworks base URL to Understudy: chat, embeddings, images, and everything else would go through it too.**

## Serve

```bash
understudy serve --lead-url https://api.fireworks.ai/inference/v1/systemone --mode shadow
```

Auth passes through: the caller's `Authorization` header is forwarded. Or have Understudy hold the key:
`--route-key-env '*=FIREWORKS_API_KEY'` (read from the environment only).

## Question types

`noul`, `choice`, `score` (Fireworks: up to 256 questions, 208 options, 2 to 10 score levels, text only; `images` is a 400).
No `box`. Rune serves this API and Decisions only: chat completions against it return a 400.

## Model name

Fireworks' [docs](https://docs.fireworks.ai/api-reference/post-systemone.md) name the model
`accounts/fireworks/models/rune`. The `accounts/fireworks/routers/rune` name also answered on this machine's integration host, but
the docs do not list it, so use the documented one.

## Tested 2026-11-02

Through `understudy serve --mode forward --route '*=<proxy for the Rune System One URL>'` (a key-injecting proxy on the test machine): one `noul` + `choice` request for each of the two model names above. Result: 200, `x-understudy: lead`, the
System One body unchanged (`urgent` 0.998, `team` technical). Not tested: learning to graduation, `serve` mode.
