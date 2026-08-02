"""Write one concrete violating transition per failed property, per design.

A witness *count* is not evidence. Table 6 of the paper reports how many
transitions break each claim, but a reader who wants to contest a verdict needs
the transition itself. This writes both a machine-readable
``results/counterexamples.json`` and a readable ``results/counterexamples.txt``.

    python dump_counterexamples.py [--limit N]

P9 is different: its witnesses are *states* with no legitimate restoration path,
not transitions, so those are listed separately.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sgrd import modelcheck as mc

RESULTS = Path(__file__).resolve().parent / "results"


def _state(s: mc.State) -> dict[str, object]:
    return {f: getattr(s, f) for f in s._fields}


def collect(limit: int) -> dict[str, object]:
    out: dict[str, object] = {}
    for name in mc.VARIANTS:
        res = mc.verify(name)
        design: dict[str, object] = {
            "states": res.n_states,
            "transitions": res.n_transitions,
            "satisfied": res.n_satisfied,
            "failed": {},
        }
        for prop, count in res.witnesses.items():
            if count == 0:
                continue
            entry: dict[str, object] = {
                "claim": mc.PROPERTY_CLAIMS[prop],
                "violating_transitions": count,
            }
            if prop == "P9":
                entry["unrecoverable_states"] = [
                    _state(s) for s in res.p9_unrecoverable[:limit]
                ]
            else:
                entry["examples"] = [
                    {"from": _state(pre), "action": act, "to": _state(post)}
                    for pre, act, post in mc.counterexample(name, prop, limit)
                ]
            design["failed"][prop] = entry
        out[name] = design
    return out


def as_text(data: dict[str, object]) -> str:
    lines = ["SGRD -- counterexamples per design", ""]
    for name, d in data.items():
        lines.append(f"== {name}: {d['satisfied']}/12 satisfied "
                     f"({d['states']} states, {d['transitions']} transitions)")
        if not d["failed"]:
            lines.append("   no violated property")
        for prop, e in d["failed"].items():
            lines.append(f"   {prop} ({e['violating_transitions']} violating): {e['claim']}")
            for ex in e.get("examples", []):
                pre = {k: v for k, v in ex["from"].items()}
                post = {k: v for k, v in ex["to"].items()}
                delta = {k: (pre[k], post[k]) for k in pre if pre[k] != post[k]}
                lines.append(f"      action={ex['action']}  changed={delta}")
                lines.append(f"      from={pre}")
            for st in e.get("unrecoverable_states", []):
                lines.append(f"      unrecoverable state={st}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=1,
                    help="examples per failed property (default 1)")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    data = collect(args.limit)
    (RESULTS / "counterexamples.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "counterexamples.txt").write_text(as_text(data), encoding="utf-8")
    print(f"wrote {RESULTS / 'counterexamples.json'}")
    print(f"wrote {RESULTS / 'counterexamples.txt'}")


if __name__ == "__main__":
    main()
