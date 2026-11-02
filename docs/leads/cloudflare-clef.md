# Cloudflare Clef and Clef Flash

**STATUS: planned.** Not usable behind `understudy serve` today: the questions and answers match System One, but the reply is
wrapped in Cloudflare's API envelope, and routing alone cannot remove it.

## What Cloudflare documents

[Workers AI](https://developers.cloudflare.com/workers-ai/models/clef/index.md): `@cf/cloudflare/clef` (27B, 65,536-token
context) and `@cf/cloudflare/clef-flash` (9B, 24,576). Endpoint:

```text
POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/cloudflare/clef
Authorization: Bearer <Workers AI token>
```

The body is System One's: `{"model": "clef" | "clef-flash", "state", "questions"}` with `noul`, `choice` and `score` (at most 64
questions, no `box`), plus an optional `images` field (a Clef extension). Cloudflare does not call it System One compatible.
Long state is truncated to fit the context after the questions and images, so a long state loses its tail, silently.

## Tested 2026-11-02

- Direct to Cloudflare: 200, with the System One answers **inside** an envelope:
  `{"result": {"model":"clef","answers":{...},"usage":{...}}, "success":true, "errors":[], "messages":[]}`.
- Through `understudy serve --mode forward --route 'clef=<the URL above>' --route-key-env clef=CLOUDFLARE_API_TOKEN`: the path and
  the token (from the environment) both worked, so a route mapping is enough for the URL and auth. But the caller received the
  envelope, not a System One body, so a System One caller (and Understudy's learning, which reads the answers) gets the wrong shape.

## What is missing

An unwrapping adapter: take `result` out of the envelope, and map a failure (`success: false`, `errors`) to an HTTP error. That is
more than a path mapping, so it is in [TODO](../../TODO.md) under "After the release". Until then, do not put Clef behind
Understudy.

## When it lands

Keep decision calls on a separate client with its URL set in code; **do not change a global Cloudflare base URL, because that
would send every Workers AI call through Understudy.**

```bash
understudy serve --lead-url "https://api.cloudflare.com/client/v4/accounts/$CLOUDFLARE_ACCOUNT_ID/ai/run/@cf/cloudflare/clef" \
  --route-key-env '*=CLOUDFLARE_API_TOKEN' --mode shadow
```
