# TypeSafe Jev

**STATUS: works today** for text questions (`noul`, `choice`, `score`). Box and image answers: see below.

Jev is the native System One lead. Understudy speaks its wire format as is, so nothing is translated.

## In the caller

TypeSafe's API *is* the decision model, so pointing its SDK at Understudy is safe:

```bash
export TYPESAFE_BASE_URL=http://127.0.0.1:8787
```

This only affects the TypeSafe SDK. Other code, and other vendors' clients, are untouched. A plain HTTP caller posts to
`http://127.0.0.1:8787/v1/systemone` with model `jev-latest`.

## Serve

```bash
understudy serve --lead-url https://api.typesafe.ai/v1/systemone --route-key-env '*=TYPESAFE_API_KEY' --mode shadow
```

The public URL and the Bearer key come from [docs.typesafe.ai](https://docs.typesafe.ai/api/system-one.md)
(`POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer <key>`). If the caller sends its own `Authorization`
header, drop `--route-key-env` and it is forwarded.

## Question types

| type | works |
|---|---|
| `noul`, `choice`, `score` | yes: forwarded, learned from, graduated |
| `box` (image localization) | **Jev-only.** Fenced off as opaque: a request containing one is forwarded whole and never learned, so it reaches Jev unchanged. Not tested end to end (below). Local box answers are planned ([TODO](../../TODO.md)) |

Requests with images or Box questions need the header `TypeSafe-Danger-Zone-Images: true`. Understudy forwards every
end-to-end header, so it passes through (by the code; not verified live). TypeSafe says Box and image support is not yet
production-ready and may change between releases.

## Tested 2026-11-02

Through `understudy serve --mode forward --route 'jev-*=<proxy for the Jev System One URL>'` (a key-injecting proxy on the test machine, no key needed): one request with a `noul` and a `choice` question, model `jev-latest`. Result: 200,
`x-understudy: lead`, the System One body unchanged (`jev-1.14.0`, `urgent` 0.97, `team` technical). Not tested: Box (my one
attempt, a 32x32 PNG, got a 400 from Jev itself, the same when posted directly, so the request was malformed rather than a proxy
fault), learning to graduation, `serve` mode.
