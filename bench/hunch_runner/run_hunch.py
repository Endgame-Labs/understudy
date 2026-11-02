"""Replay an understudy-bench stream through the REAL hunch proxy (undrgrnd-labs/hunch@22cf730, `hunch.Proxy`).

    uv run --project bench/hunch_runner python bench/hunch_runner/run_hunch.py \
        --question q.json --stream stream.jsonl --out decisions.hunch.jsonl
    # warm start (split mode): seed with train.jsonl's lead answers, then answer operate.jsonl's items
    ... --train-examples train.jsonl --operate operate.jsonl

The lead is a replay backend: for each request it returns the current stream item's `lead_probs` as a
systemone answer (`hunch.proxy.answer_object`). Determinism: refits run synchronously (`background=False`), the
proxy's audit RNG is seeded (`seed`, default 0), Python's and NumPy's global RNGs are seeded, label asks are off
(there are no labels), and after every request the runner calls `proxy.wait()`, so the background audit (hunch
always audits on a thread) finishes before the next item. The store is a SQLite file in a temporary directory.
hunch ignores question.json's `pair` (it reads the whole state as one text). hunch's own defaults are kept
(layer 8, audit 0.10, target 0.90, minority 0.80, refit_every 100, min_examples 60, memo on).

Decision fields (SPEC.md), and which are exact or approximate:
  answered_by    exact: hunch "proxy" -> "understudent"; "backend" -> "lead"; "cache" (memo: a state hunch already has
                 a lead answer for, i.e. a duplicate state) -> "lead". hunch's own value is kept in
                 "hunch_answered_by".
  label, probs   exact: parsed from the answer the caller got.
  understudent_label, understudent_probs, eligible, mode, demoted
                 exact: `Student.predict` and `Student.mode/demoted` with the fit in place just before the request,
                 which is the fit hunch's own routing uses for that request (nothing refits in between). The runner
                 calls them on the same features first (hunch's encoder cache makes hunch's own encode a hit).
                 understudent_label/probs are null until the understudent has a fit with two or more classes.
  novel          approximation: recomputed as in `Student.predict` (cosine NN among the fit's reference vectors below
                 its 5th-percentile floor), against the serving fit when graduated, else against the latest fit;
                 hunch itself only checks novelty for a graduated fit.
  audited        exact for local answers: true when hunch sent this locally answered item to the lead as an audit
                 (its `audited` counter moved). Forwarded items that audit the understudent are not flagged here; the
                 report counts them from `eligible`.
  readout        always "mean" (hunch's student is the layer-8 token mean + logistic regression).
  understudent_us     approximation: wall time of `proxy.features([text])` (one item, uncached) plus `Student.predict`, as
                 measured by this runner, not inside hunch.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from hunch.proxy import Proxy, answer_object, parse_answer, state_text


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def novelty(st, x) -> bool:
    f, _ = st._serving()
    if f is None:
        f = st.fit
    if f.ref is None or len(f.ref) < 2:
        return False
    xn = (x / (np.linalg.norm(x) + 1e-12)).astype(np.float32)
    return bool(float((f.ref @ xn).max()) < f.nn_floor)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--question", required=True)
    ap.add_argument("--stream", required=True, help="stream.jsonl: {id, state, lead_probs} (the replayed lead)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-examples", help="split mode: {id, state, lead_probs} lines given to Proxy.train as sampled")
    ap.add_argument("--operate", help="split mode: {id, state} lines to answer (default: the whole stream)")
    ap.add_argument("--layer", type=int, default=8)
    ap.add_argument("--audit-rate", type=float, default=0.10)
    ap.add_argument("--target", type=float, default=0.90)
    ap.add_argument("--minority", type=float, default=0.80)
    ap.add_argument("--min-examples", type=int, default=60, help="hunch's default is 60")
    ap.add_argument("--refit-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-memo", action="store_true", help="don't answer repeated states from the store")
    args = ap.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    qf = json.loads(Path(args.question).read_text())
    q, qid, k = qf["question"], qf["name"], len(qf["options"])
    stream = read_jsonl(args.stream)
    lead = {s["id"]: s["lead_probs"] for s in stream}
    if args.operate:
        todo = read_jsonl(args.operate)
    else:
        todo = [{"id": s["id"], "state": s["state"]} for s in stream]

    current = {"id": None}
    calls = {"n": 0}

    def backend(request):
        calls["n"] += 1
        p = np.asarray(lead[current["id"]], dtype=float)
        return {"model": "replay", "answers": {qid: answer_object(q, p)}}

    import eg2cpu

    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="hunch-runner-") as tmp:
        proxy = Proxy(eg2cpu.Model(), backend=backend, db=str(Path(tmp) / "proxy.db"), layer=args.layer,
                      audit=args.audit_rate, target=args.target, minority=args.minority,
                      refit_every=args.refit_every, min_examples=args.min_examples, background=False,
                      memo=not args.no_memo, ask_for_labels=False, seed=args.seed)
        st = proxy._student(q)
        summary = {"system": "hunch", "items": len(todo)}
        if args.train_examples:
            train = read_jsonl(args.train_examples)
            t0 = time.perf_counter()
            status = proxy.train(q, [{"state": r["state"], "answer": [float(x) for x in r["lead_probs"]],
                                      "kind": "sampled"} for r in train], refit=True)
            proxy.wait()
            summary |= {"train_examples": len(train), "train_s": round(time.perf_counter() - t0, 3),
                        "graduated_after_train": bool(status["graduated"]), "mode_after_train": status["mode"],
                        "report_after_train": status["report"]}
            print(json.dumps({"after_train": {k_: summary[k_] for k_ in ("train_examples", "train_s",
                                                                           "graduated_after_train", "mode_after_train")}}),
                  file=sys.stderr, flush=True)

        first_local = None
        n_local = 0
        with open(args.out, "w") as out:
            for i, item in enumerate(todo):
                state = item["state"]
                text = state_text(state)
                mode, demoted = st.mode, st.demoted
                t0 = time.perf_counter()
                _, F = proxy.features([text])
                x = F[0]
                p, eligible, local, _tier = st.predict(x)
                us = (time.perf_counter() - t0) * 1e6
                fitted = st.fit.coef is not None
                novel = novelty(st, x)
                audited_before = proxy.stats["audited"]
                current["id"] = item["id"]
                got = proxy.answer({"state": state, "questions": {qid: q}})
                if not proxy.wait(timeout=600):
                    raise RuntimeError("hunch did not settle (refit/audit) within 600 s")
                by = got["answered_by"]
                ans = got["answers"][qid]
                label, probs = parse_answer(q, {kk: v for kk, v in ans.items() if kk not in ("answered_by", "tier")})
                if probs is None:
                    probs = np.eye(k)[label].tolist()
                understudent = by == "proxy"
                if understudent and not local:
                    print(f"warning: {item['id']}: hunch answered locally but the pre-request predict said not local",
                          file=sys.stderr)
                if understudent:
                    n_local += 1
                    if first_local is None:
                        first_local = i + 1
                rec = {"id": item["id"], "answered_by": "understudent" if understudent else "lead",
                       "label": int(label), "probs": [round(float(v), 6) for v in probs],
                       "understudent_label": int(np.argmax(p)) if fitted else None,
                       "understudent_probs": [round(float(v), 6) for v in p] if fitted else None,
                       "eligible": bool(eligible), "novel": novel, "demoted": bool(demoted), "mode": mode,
                       "audited": bool(understudent and proxy.stats["audited"] > audited_before),
                       "readout": "mean", "understudent_us": round(us, 1), "hunch_answered_by": by}
                out.write(json.dumps(rec) + "\n")
                if (i + 1) % 100 == 0:
                    print(f"{i + 1}/{len(todo)} items, {n_local} local, mode {st.mode}, demoted {st.demoted}",
                          file=sys.stderr, flush=True)
        final = proxy.question_status(st.key)
        summary |= {"first_local_item": first_local, "local": n_local,
                    "coverage": round(n_local / max(1, len(todo)), 4), "lead_calls": calls["n"],
                    "stats": proxy.status()["stats"], "final_mode": final["mode"], "final_graduated": final["graduated"],
                    "final_report": final["report"], "audits": final["audits"],
                    "wall_s": round(time.perf_counter() - t_start, 3)}
        proxy.close()
    print(json.dumps(summary, default=str), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
