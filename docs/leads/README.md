# Lead setup

A **lead** is the decision model Understudy sits in front of ([api: Leads](../api.md#leads)). One page per lead says what the
caller sets, the `understudy serve` command, how auth passes through, which question types work, and how far it was tested.

| lead | page | status |
|---|---|---|
| TypeSafe **Jev** | [typesafe-jev.md](typesafe-jev.md) | works today (text questions); Box/image answers not tested end to end |
| Fireworks **Rune** | [fireworks-rune.md](fireworks-rune.md) | works today |
| Cloudflare **Clef** / **Clef Flash** | [cloudflare-clef.md](cloudflare-clef.md) | planned: the reply is wrapped in an envelope |
| OpenAI **Decisions** (`gpt-6-luna`) | [openai-decisions.md](openai-decisions.md) | planned: a different request and response shape |

"Works today" means a few real calls went through `understudy serve` and came back as System One, on the date on each page. It
does not mean a benchmark; see [benchmarks](../benchmarks.md) for those.

## Caller rule

Understudy answers one thing: the decision call. **Never change a global base URL** (`OPENAI_BASE_URL`, a global Fireworks or
Cloudflare base URL) to point at Understudy: chat, embeddings, images and everything else would go through it too, and it
answers none of them. Make a separate client used only for decision calls, with its `base_url` set in code. Only the TypeSafe
SDK is different, because TypeSafe's API is the decision model.

## Serve command

Each page uses `--lead-url`, the catch-all route:

```bash
understudy serve --lead-url <the lead's decision endpoint> [--route-key-env ...] --mode shadow --snapshot-dir snapshots/
```

Start in `shadow` (learn from the lead, answer nothing locally), then move to `serve` once questions have graduated
([graduation](../graduation.md)). `serve --help` lists the older, equivalent `--route 'MODEL=URL'` (or `'*=URL'`) form, which
is what the tests on these pages used. Never point a live production caller at `understudy serve` except in shadow or replay.

## Auth

A route's key comes from an environment variable named by `--route-key-env NAME_OR_MODEL=ENV_VAR` and is sent upstream as
`Authorization: Bearer <value>`. Without it, the caller's own `Authorization` header is forwarded. A lead that injects
credentials itself needs neither. The key is never read from a file or argv.
