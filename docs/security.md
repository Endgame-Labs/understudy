# Security and privacy

What Understudy sees, what it writes to disk, where it listens, and how it handles keys. Read this before pointing real
traffic at it. For the modes referred to here see [how it works](how-it-works.md) and the [API](api.md).

- [What it sees](#what-it-sees)
- [What it writes to disk](#what-it-writes-to-disk)
- [What it sends to the lead](#what-it-sends-to-the-lead)
- [The listener](#listener)
- [Keys](#keys)
- [Recommended setup](#recommended-setup)

## What it sees

`understudy serve` is a proxy, so it sees every prompt and state in full, and every answer the lead returns. It parses them to
learn the question, featurize the state, and compare answers. Treat the process, its memory, and its files with the same care
as the traffic itself. Understudy makes no attempt to redact states.

## What it writes to disk

| file | what it holds | when it is written |
|---|---|---|
| the **capture log** (`--capture FILE`) | one JSON line per request: sizes, timings, a hash of the state, the lead's and the understudent's answers per question. With `--capture-bodies` it also holds the full request and response bodies, states included | only when `--capture` is given |
| a **snapshot** (`--snapshot-dir DIR`) | per question, up to 2,000 recent examples **with their state text**, the lead's answers, the fitted models, and the audit window, so a restart does not forget what was learned | only when `--snapshot-dir` is given |
| the **rules cache** (`rules_llm.jsonl`, `--rules-cache`) | every request and response of the `rules` design's chat model, which include labeled example states | only when `rules` is in use |

Understudy does not set file modes itself: files follow the process umask (typically readable by other local users). Create the
directories yourself, readable only by the service user, and run with `umask 077`. Delete a snapshot when you retire a
question, and do not commit capture logs or snapshots: a snapshot is a copy of your traffic.

## What it sends to the lead

- Every forwarded request goes to the lead exactly as the caller sent it, because that is what the caller asked for.
- **Audits** send a sample of locally answered states to the lead (about one in ten, `--audit-rate`) so it can check the
  understudent. Those states leave your boundary even though the caller got a local answer. If the lead is not allowed to see
  some states, a graduated question is not the place to economize: keep that route in `forward` mode or do not route it
  through Understudy.
- The `rules` design sends up to 40 labeled example states to its chat model to write regexes. Point `--rules-llm-url` at a
  model you trust with that text, or do not use `rules`.
- Nothing is sent anywhere else: no telemetry, no update check.

## Listener

- `serve` binds `127.0.0.1:8787` by default (`--bind`) and has **no authentication**. Anyone who can reach the port can send
  requests through it with the lead's credentials, and can read `/status`.
- Do not bind a public or shared interface. If another machine must reach it, put it behind a network boundary or an
  authenticating reverse proxy that you control.
- Requests are limited in size, header count, connections, and time (see [limits](api.md#limits)), but those limits are
  protections against accidents, not a defence against a hostile network.

## Keys

- Upstream keys are read from **environment variables** (`--route-key-env ROUTE=ENV_VAR`, `--lead-key-env`, `key_env` in the
  config file). The config file names the variable, never the key.
- Keys are not written to the capture log, the snapshot, or the status page.
- A caller's own `Authorization` header is not used for the lead: the keys are the lead's, held by Understudy.

## Recommended setup

1. Run `serve` on loopback, as a user that owns nothing else, with `umask 077`.
2. Start in `--mode shadow` with `--capture` into a private directory and read the capture log before letting a question answer locally (the default `serve` mode answers a question once it graduates).
3. Leave `--capture-bodies` off unless you are debugging, and delete the log afterwards.
4. Keep snapshot directories private and out of version control.
5. Report a vulnerability to the maintainer privately rather than in a public issue.
