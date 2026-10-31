#!/usr/bin/env python3
"""Run the reference-implementation experiments of Section `sec:impl`.

    python run_refimpl.py                 # tiers 0 and 1, ~2 minutes
    python run_refimpl.py --quick         # fewer timing repetitions
    python run_refimpl.py --no-tpm        # tier 0 only
    python run_refimpl.py --strict        # exit non-zero if any check fails

**Tier 0** (S1-S5) needs nothing but a Python interpreter. **Tier 1** (T1-T5)
needs a TPM 2.0, which every machine meeting the Windows 11 installation
requirement has; it is skipped, not failed, where there is none.

Nothing here changes persistent state on the machine. Tier 1 creates keys under
the TPM's null hierarchy and deletes the CNG keys it makes; it defines no NV
index, takes no ownership, and writes no PCR.

Outputs land in ``results/``:

    refimpl.json   every check and timing, machine-readable
    refimpl.txt    the same as a printable report

The only dependency beyond the standard library is ``cryptography`` (for
AES-GCM, HKDF, and the independent signature check in T1); the group arithmetic
is plain ``int`` and the TPM access is ``ctypes`` over the Windows interfaces.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from sgrd.refimpl import experiments as ex
from sgrd.refimpl import protocol as pr

HERE = Path(__file__).resolve().parent
RULE = "-" * 88

# Section `sec:results`. The latency budget is only interesting against the
# timescale it is being compared with, so the comparison is made explicit rather
# than left for the reader to reconstruct.
CTMC_MEAN_RESTORATION_H = 23.0
CTMC_QUORUM_RATE_PER_H = 0.25


def render(res: ex.Results) -> str:
    lines: list[str] = []
    add = lines.append

    add(RULE)
    add("SGRD tier-0 reference implementation -- Section sec:impl, experiments "
        "S1-S5")
    add(RULE)
    add("")
    add("Parameters")
    for k, v in res.parameters.items():
        add(f"  {k:<14} {v}")
    add("")
    add("Group self-check (the embedded RFC 3526 constant, verified not "
        "assumed)")
    for k, v in res.group_self_check.items():
        add(f"  {k:<20} {'ok' if v else 'FAILED'}")

    def render_checks(checks: list[ex.Check]) -> None:
        for c in checks:
            mark = "  ok  " if c.passed else " FAIL "
            add(f"{mark}{c.name}")
            got = c.observed + (f" [{c.detail}]" if c.detail else "")
            if c.passed:
                add(f"        -> {got}")
            else:
                add(f"        expected {c.expected}, observed {got}")
            if c.note:
                for chunk in _wrap(c.note, 78):
                    add(f"        {chunk}")

    add("")
    add(RULE)
    add("TIER 0 -- no security hardware")
    add(RULE)
    for key, (title, _) in ex.EXPERIMENTS.items():
        checks = [c for c in res.checks if c.experiment == key]
        n_ok = sum(c.passed for c in checks)
        add("")
        add(RULE)
        add(f"{key}  {title}")
        add(f"    {n_ok}/{len(checks)} checks passed")
        add(RULE)
        render_checks(checks)

    crash = [c for c in res.checks if c.experiment == "A"]
    if crash:
        from sgrd.refimpl import crash_experiments as cx
        add("")
        add(RULE)
        add("CRASH CONSISTENCY OF THE CAPSULE COMMIT")
        add(f"    {sum(c.passed for c in crash)}/{len(crash)} checks passed")
        add(RULE)
        for key, (title, _) in cx.CRASH_EXPERIMENTS.items():
            group = [c for c in crash if c.name.startswith(key + " ")]
            if not group:
                continue
            add("")
            add(f"{key}  {title}")
            render_checks(group)
        add("")
        add("  What the medium is assumed to provide, and this does not check:")
        from sgrd.refimpl import store as _st
        for req in _st.MEDIUM_REQUIREMENTS:
            for k, chunk in enumerate(_wrap(req, 82)):
                add(f"    {'-' if k == 0 else ' '} {chunk}")

    add("")
    add(RULE)
    add("S6  Latency budget of the recovery path")
    add(RULE)
    add(f"{'operation':<46}{'n':>4}{'mean ms':>10}{'median ms':>11}"
        f"{'p95 ms':>10}")
    for t in res.timings:
        add(f"{t.operation:<46}{t.n:>4}{t.mean_ms:>10.2f}{t.median_ms:>11.2f}"
            f"{t.p95_ms:>10.2f}")
    total = sum(t.mean_ms for t in res.timings
                if not t.operation.startswith("full episode"))
    episode = next((t.mean_ms for t in res.timings
                    if t.operation.startswith("full episode")), 0.0)
    ctmc_ms = CTMC_MEAN_RESTORATION_H * 3.6e6
    ratio = episode / ctmc_ms
    orders = -math.log10(ratio) if ratio > 0 else float("inf")
    add("")
    add(f"  sum of the parts          {total / 1000:8.3f} s")
    add(f"  measured end-to-end       {episode / 1000:8.3f} s")
    add(f"  quorum response time      {1 / CTMC_QUORUM_RATE_PER_H:8.0f} h"
        f"   (1/rho, Section sec:results)")
    add(f"  mean restoration time     {CTMC_MEAN_RESTORATION_H:8.0f} h"
        f"   = {ctmc_ms / 1000:,.0f} s")
    add(f"  the exchange as a fraction of it         {ratio:.2e}"
        f"   ({orders:.1f} orders of magnitude below)")
    add("")
    for chunk in _wrap(
            "These are CPython `pow` on a 2048-bit modulus and are therefore a "
            "loose upper bound -- a tuned library would be roughly an order of "
            "magnitude faster and hardware more again. The looseness runs in "
            "the safe direction: even at this bound the whole cryptographic "
            f"exchange is {orders:.1f} orders of magnitude below the "
            "quorum-gathering time that dominates the "
            f"{CTMC_MEAN_RESTORATION_H:.0f}-hour mean, so no plausible change "
            "to these operations moves any quantitative claim in Section "
            "sec:results. The converse is why it is worth measuring at all: "
            "had they come out in hours, the CTMC parameterisation would need "
            "redoing.", 86):
        add(f"  {chunk}")

    # --- tier 1 -------------------------------------------------------------
    tpm = res.hardware.get("tpm", {}) if res.hardware else {}
    pcp = res.hardware.get("platform_crypto_provider", {}) if res.hardware else {}
    tier1 = [c for c in res.checks if c.experiment == "T"]

    add("")
    add(RULE)
    add("TIER 1 -- TPM 2.0")
    add(RULE)
    if not tier1:
        add("  not run.")
        for name, d in (("tpm", tpm), ("platform crypto provider", pcp)):
            add(f"    {name}: {d.get('reason', d) if d else 'not probed'}")
        for chunk in _wrap(
                "Tier 0 above is unaffected. The tier-1 experiments are skipped "
                "rather than failed on a machine without a usable TPM.", 86):
            add(f"  {chunk}")
    else:
        add(f"  TPM      {tpm.get('manufacturer', '?')} "
            f"{tpm.get('vendor_string', '?')}, "
            f"firmware {tpm.get('firmware_version', '?')}, "
            f"spec rev {tpm.get('spec_revision', '?')}")
        add(f"  provider {pcp.get('provider', '?')} "
            f"({pcp.get('algorithm_group', '?')})")
        add(f"  reset count {tpm.get('reset_count', '?')}, "
            f"clock safe {tpm.get('clock_safe', '?')}, "
            f"application PCR unextended "
            f"{tpm.get('application_pcr_unextended', '?')}")
        from sgrd.refimpl import tpm_experiments as tx
        for key, (title, _) in tx.TPM_EXPERIMENTS.items():
            group = [c for c in tier1 if c.name.startswith(key + " ")]
            add("")
            add(RULE)
            add(f"{key}  {title}")
            add(f"    {sum(c.passed for c in group)}/{len(group)} checks passed")
            add(RULE)
            render_checks(group)
        rest = [c for c in tier1 if c.name.startswith("T5 ")]
        if rest:
            add("")
            add(RULE)
            add("T5  The tier-0 exchange driven by real hardware inputs")
            add(f"    {sum(c.passed for c in rest)}/{len(rest)} checks passed")
            add(RULE)
            render_checks(rest)
        add("")
        for chunk in _wrap(
                "What tier 1 does not establish: a TPM is not a PUF, so nothing "
                "here speaks to physical unclonability; PCRs 0-7 measure the "
                "boot chain and not the autonomy stack; and the TPM clock is "
                "monotonic hardware but it is not the design's counter, which "
                "wants an NV index of type TPM_NT_COUNTER and therefore owner "
                "authorisation. Zeroisation completeness and power-loss "
                "atomicity remain untouched.", 86):
            add(f"  {chunk}")

    add("")
    add(RULE)
    add(f"TOTAL  {res.n_passed}/{len(res.checks)} checks passed"
        f"{'' if res.all_passed else '  -- SOME CHECKS FAILED'}")
    add(RULE)
    return "\n".join(lines)


def _wrap(text: str, width: int) -> list[str]:
    words, out, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        out.append(cur)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="fewer timing repetitions")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any check fails")
    ap.add_argument("--no-tpm", action="store_true",
                    help="skip the tier-1 experiments even if a TPM is present")
    ap.add_argument("--outdir", default=str(HERE / "results"))
    ap.add_argument("-t", "--threshold", type=int, default=3)
    args = ap.parse_args(argv)

    res = ex.run_all(t=args.threshold, orgs=pr.DEFAULT_ORGS,
                     timing_n=4 if args.quick else 20,
                     tier1=not args.no_tpm)
    report = render(res)
    print(report)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "refimpl.txt").write_text(report + "\n", encoding="utf-8")
    (outdir / "refimpl.json").write_text(
        json.dumps(res.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"\nwritten to {outdir / 'refimpl.txt'} and {outdir / 'refimpl.json'}")

    return 1 if args.strict and not res.all_passed else 0


if __name__ == "__main__":
    sys.exit(main())
