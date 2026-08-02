#!/usr/bin/env python3
"""Reproduce every number in the SGRD paper from the published parameterisation.

    python run_all.py                      # full reproduction + figures
    python run_all.py --asam-csv asam.csv  # also re-derive Table 3 from raw data
    python run_all.py --no-figures         # skip matplotlib (numpy only)
    python run_all.py --strict             # exit non-zero on any discrepancy

Results land in ``results/`` as JSON and CSV; figures in ``results/figures/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sgrd import asam, ctmc, modelcheck, report
from sgrd import params as P
from sgrd.threshold import p_authority_assemblable, p_unauthorised_restoration

HERE = Path(__file__).resolve().parent
RULE = "-" * 96


def _h(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def section_asam(csv_path: str | None, outdir: Path) -> dict:
    _h("Section VI-C / Table 3 -- possession duration from NGA ASAM")
    print(f"{'Stratum':<32}{'n':>4}{'median':>9}{'mean':>9}{'p90':>9}{'max':>9}")
    published = {}
    for s in asam.published_table():
        print(f"{s.name:<32}{s.n:>4}{s.median_h:>9.0f}{s.mean_h:>9.0f}"
              f"{s.p90_h:>9.0f}{s.max_h:>9.0f}")
        published[s.name] = dict(n=s.n, median_h=s.median_h, mean_h=s.mean_h,
                                 p90_h=s.p90_h, max_h=s.max_h)

    problems = asam.check_internal_consistency()
    print("\ninternal consistency:",
          "strata partition the pooled sample and the means reconcile"
          if not problems else "; ".join(problems))

    out = {"published": published, "consistency_problems": problems}

    if csv_path:
        print(f"\nre-deriving from {csv_path} ...")
        try:
            audit = asam.derive_from_export(csv_path)
        except (OSError, ValueError) as exc:
            print(f"  could not derive: {exc}")
            out["derivation_error"] = str(exc)
        else:
            print(f"  {audit.n_records} incident records read")
            print(f"  {audit.n_possession_events} possession-type events "
                  f"identified from narratives")
            print(f"  {len(audit.candidates)} candidate durations "
                  f"(a release date was parseable)")
            print(f"  {len(audit.removed)} removed by audit, "
                  f"{len(audit.kept)} retained")
            for obs, reason in audit.removed:
                print(f"    - {obs.reference}: {reason}")
            audit_csv = outdir / "asam_audit.csv"
            asam.write_audit_csv(audit, audit_csv, source_path=csv_path)
            print(f"  per-observation audit trail written to {audit_csv}")
            out["audit_csv"] = str(audit_csv)
            if audit.kept:
                derived = asam.summarise(audit.kept, "Pooled (re-derived)")
                print(f"\n  {derived.name:<30}{derived.n:>4}"
                      f"{derived.median_h:>9.0f}{derived.mean_h:>9.0f}"
                      f"{derived.p90_h:>9.0f}{derived.max_h:>9.0f}")
                out["derived_pooled"] = derived.__dict__
    else:
        print("\n(no --asam-csv given: the quantitative model needs only the "
              "stratum means, which are published)")

    print("\ncaveats carried with these numbers:")
    for i, c in enumerate(asam.CAVEATS, 1):
        print(f"  {i}. {c}")
    return out


def section_modelcheck() -> dict:
    _h("Section VII-A / Table 6 -- exhaustive property verification")
    out = {}
    names = ("B0", "B1", "B2", "B3", "H1")
    results = {name: modelcheck.verify(name) for name in names}

    print(f"{'':<5}" + "".join(f"{n:>7}" for n in names) + "   claim")
    for prop in (f"P{i}" for i in range(1, 13)):
        row = "".join(f"{results[n].witnesses[prop]:>7}" for n in names)
        print(f"{prop:<5}{row}   {modelcheck.PROPERTY_CLAIMS[prop]}")
    print(f"{'':<5}" + "-" * (7 * len(names)))
    print(f"{'sat':<5}"
          + "".join(f"{results[n].n_satisfied:>4}/12" for n in names))

    for name, res in results.items():
        ref = P.PUBLISHED["modelcheck"][name]
        print(f"\n{name}: {res.n_states} reachable states, "
              f"{res.n_transitions} transitions "
              f"(paper: {ref['states']} / {ref['transitions']})")
        out[name] = {
            "states": res.n_states,
            "transitions": res.n_transitions,
            "satisfied": res.n_satisfied,
            "witnesses": res.witnesses,
        }

    # A witness count is not evidence; print one concrete violating transition
    # per failed property so a reviewer can read what actually goes wrong.
    print("\nB0 counterexamples (one violating transition per failed property):")
    for prop, count in results["B0"].witnesses.items():
        if count == 0 or prop == "P9":
            continue
        pre, label, post = modelcheck.counterexample("B0", prop)[0]
        print(f"  {prop}: --{label}--> "
              f"[phase {pre.phase}->{post.phase}, auth={pre.auth}, "
              f"possessed={pre.possessed}, safe={pre.safe}, "
              f"attested={pre.attested}, tampered={pre.tampered}, "
              f"used={pre.used}]")

    # A 12/12 verdict says the design is sound; it does not say which part of it
    # does the work. Flip each guard in turn and report what breaks.
    print("\nAblation -- which guard discharges which property (H1):")
    abl = modelcheck.ablate("H1")
    for flag, props in abl.items():
        ref = tuple(P.PUBLISHED["ablation"].get(flag, ()))
        mark = "" if tuple(props) == ref else "   <-- differs from paper"
        print(f"  {flag:<28}{', '.join(props) if props else '(nothing breaks)'}"
              f"{mark}")
    load_bearing = sum(1 for p in abl.values() if p)
    print(f"  {load_bearing} of {len(abl)} guards are load-bearing in the model; "
          f"the rest are design\n  requirements for reasons the model does not "
          f"represent.")
    out["ablation"] = {k: list(v) for k, v in abl.items()}
    return out


def section_extraction() -> dict:
    _h("Section VII-B -- phase one: what possession decides (independent of t)")
    regimes = [("Pooled", P.POSSESSION_POOLED_H)]
    regimes += sorted(P.POSSESSION_STRATA_H.items(), key=lambda kv: kv[1])

    priced = sorted(P.TIME_TO_EXTRACTION_H)
    cols = [f"{v}/{a}" for v in priced for a in ("A2", "A3")]
    print(f"{'Regime':<32}{'mean (h)':>10}" + "".join(f"{c:>9}" for c in cols))
    out = {}
    for name, hours in regimes:
        row = {f"{v}/{a}": ctmc.extraction_probability(v, a, hours)
               for v in priced for a in ("A2", "A3")}
        print(f"{name:<32}{hours:>10.0f}"
              + "".join(f"{row[k]:>9.3f}" for k in cols))
        out[name] = {"mean_possession_h": hours, **row}

    b0 = [v["B0/A2"] for k, v in out.items() if k != "Pooled"]
    print(f"\nB0 exposure varies {max(b0) / min(b0):.1f}x across regions "
          f"({min(b0):.3f} to {max(b0):.3f}) -- set by possession duration, "
          f"which no design controls")
    print("H1 is rated, not priced: the attack-potential worksheet places its "
          "remaining path\nin a band (Section VI-D). No extraction probability "
          "is claimed for it.")
    return out


def section_ctmc() -> dict:
    _h("Section VII-C / Table 7 -- phase two: what the threshold decides")
    print(f"{'Configuration':<28}{'var':>4}{'P(ext)':>9}{'P(rest|c)':>11}"
          f"{'P(loss|c)':>11}{'E[T_c] h':>10}{'P_u':>11}")
    out = {}
    for cfg in P.named_configurations():
        o = ctmc.evaluate(cfg, adversary="A2")
        pe = f"{o.p_extracted:.3f}" if o.p_extracted is not None else "rated"
        print(f"{cfg.name:<28}{cfg.variant:>4}{pe:>9}"
              f"{o.p_restored_given_custody:>11.3f}"
              f"{o.p_loss_given_custody:>11.3f}"
              f"{o.mean_time_to_restore_given_custody:>10.1f}"
              f"{o.p_unauthorised:>11.1e}")
        out[f"{cfg.name} [{cfg.variant}]"] = o.as_row()

    _h("Findings")
    print("Finding 1 -- widen n before raising t:")
    for base, wide, t in ((("Q1", "Q2", "Q3"), ("Q1", "Q2", "Q3", "Q4"), 2),
                          (("Q1", "Q2", "Q3", "Q4"),
                           ("Q1", "Q2", "Q3", "Q4", "Q5"), 3)):
        a = ctmc.evaluate(P.Configuration("a", t, P.holder_set(*base), "H1"))
        b = ctmc.evaluate(P.Configuration("b", t, P.holder_set(*wide), "H1"))
        # Post-custody dominance only. Widening n always raises P_u, so this
        # is never dominance outright -- the label says which side it is on.
        dom = (b.p_restored_given_custody > a.p_restored_given_custody
               and b.p_loss_given_custody < a.p_loss_given_custody
               and b.mean_time_to_restore_given_custody
               < a.mean_time_to_restore_given_custody)
        verdict = ("dominates post-custody, at a cost in P_u" if dom
                   else "does not dominate post-custody")
        print(f"  {t}-of-{len(base)} -> {t}-of-{len(wide)}: "
              f"P(rest|c) {a.p_restored_given_custody:.3f}->"
              f"{b.p_restored_given_custody:.3f}, "
              f"P(loss|c) {a.p_loss_given_custody:.3f}->"
              f"{b.p_loss_given_custody:.3f}, "
              f"E[T_c] {a.mean_time_to_restore_given_custody:.1f}->"
              f"{b.mean_time_to_restore_given_custody:.1f}, "
              f"P_u {a.p_unauthorised:.1e}->{b.p_unauthorised:.1e}   "
              f"{verdict}")

    print("\nFinding 2 -- the viable band over the full five-holder set:")
    print(f"  {'t':>2}{'P(D_a>=t)':>11}{'P_u':>11}{'P(loss|c)':>11}{'E[T_c] h':>10}")
    sweep = {}
    for t in range(1, len(P.HOLDERS) + 1):
        o = ctmc.evaluate(P.Configuration(f"{t}-of-5", t, P.HOLDERS, "H1"))
        pa = p_authority_assemblable(P.HOLDERS, t)
        print(f"  {t:>2}{pa:>11.3f}{o.p_unauthorised:>11.1e}"
              f"{o.p_loss_given_custody:>11.3f}"
              f"{o.mean_time_to_restore_given_custody:>10.1f}")
        sweep[t] = {"P_available": pa, "P_u": o.p_unauthorised,
                    "P_loss_given_custody": o.p_loss_given_custody,
                    "E_Tc": o.mean_time_to_restore_given_custody}
    out["sweep_five_holders"] = sweep

    print("\nFinding 3 -- the competing-risks artefact:")
    cfg = next(c for c in P.named_configurations() if c.variant == "B0")
    o = ctmc.evaluate(cfg)
    h1 = ctmc.evaluate(next(c for c in P.named_configurations()
                            if c.variant == "H1" and c.name.startswith("2-of-4")))
    survives = 1.0 - o.p_extracted
    print(f"  On post-custody metrics alone the baseline wins:")
    print(f"    P(loss|c)  B0 {o.p_loss_given_custody:.3f}   "
          f"H1 (2-of-4) {h1.p_loss_given_custody:.3f}")
    print(f"    E[T_c|OK]  B0 {o.mean_time_to_restore_given_custody:.1f} h "
          f"  H1 (2-of-4) {h1.mean_time_to_restore_given_custody:.1f} h")
    print(f"  It wins because it kept the secret it was supposed to erase, so it")
    print(f"  restores instantly -- in the {survives:.3f} of episodes where there is")
    print(f"  anything left to restore. The other {o.p_extracted:.3f} went to extraction.")
    print(f"  Unconditional loss makes the same point as an identity:")
    print(f"    P(custody regained) x P(loss|custody) = "
          f"{survives:.3f} x {o.p_loss_given_custody:.3f} = "
          f"{o.p_loss_uncond:.4f}")
    print("  Any design that leaks less carries more mass into the post-custody phase")
    print("  and so reports MORE loss there, whatever its merits. A post-custody or")
    print("  unconditional figure read alone rewards the design that leaks fastest.")
    out["finding3_identity"] = {
        "p_ext": o.p_extracted,
        "p_survives": survives,
        "p_loss_given_custody": o.p_loss_given_custody,
        "p_loss_uncond": o.p_loss_uncond,
    }
    return out


def section_reproduction(strict: bool) -> tuple[dict, bool]:
    _h("Reproduction check -- recomputed values against the printed article")
    rep = report.run()
    for c in rep.checks:
        print(c)
    print(f"\n{rep.summary()}")
    if rep.notes:
        print("\nnotes:")
        for n in rep.notes:
            print(f"  * {n}")
    ok = not rep.failures
    if not ok:
        print(f"\n{len(rep.failures)} discrepancies:")
        for c in rep.failures:
            print(f"  {c.section} :: {c.quantity}")
    return ({"summary": rep.summary(),
             "checks": [c.__dict__ for c in rep.checks],
             "notes": rep.notes}, ok or not strict)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asam-csv", help="path to a downloaded NGA ASAM bulk export")
    ap.add_argument("--outdir", default=str(HERE / "results"))
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--sensitivity", action="store_true",
                    help="rank the parameters by elasticity and report how far "
                         "each can move before a stated finding flips")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any published value fails to reproduce")
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = {
        "asam": section_asam(args.asam_csv, outdir),
        "modelcheck": section_modelcheck(),
        "extraction": section_extraction(),
        "ctmc": section_ctmc(),
    }
    results["reproduction"], ok = section_reproduction(args.strict)

    if args.sensitivity:
        from sgrd import sensitivity
        _h("Parameter sensitivity -- which stipulated values actually matter")
        print(sensitivity.format_report())
        results["sensitivity"] = {
            "elasticities": sensitivity.elasticity_table(),
            "breaking_factors": [
                {k: (list(v) if isinstance(v, tuple) else v) for k, v in row.items()}
                for row in sensitivity.breaking_factors()],
        }

    if not args.no_figures:
        _h("Figures")
        try:
            from sgrd import figures
            from sgrd.opconcept import figure_opconcept
        except ImportError as exc:
            print(f"matplotlib unavailable ({exc}); skipping figures")
        else:
            paths = figures.generate_all(outdir / "figures")
            paths.append(figure_opconcept(outdir / "figures"))
            for path in paths:
                print(f"  wrote {path}")

    json_path = outdir / "results.json"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {json_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
