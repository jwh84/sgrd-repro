"""One-at-a-time sensitivity of the headline results to each parameter.

Section `sec:disc` says the extraction results "depend on a single ratio -- time
an adversary needs, over time an adversary has". That is true of P(ext), but it
is not the whole parameter table, and it is not obvious by inspection which of
the eleven stipulated values actually move a conclusion.

This module measures it. For each parameter it perturbs the value by a factor
and reports the elasticity of four outputs:

    E = d ln(output) / d ln(input)

evaluated as a symmetric log-difference over the perturbation, so an elasticity
of 1 means "a 10% error in this input is a 10% error in that output" and an
elasticity near 0 means the input does not matter. Elasticity is unitless, so
rates, durations and probabilities are directly comparable -- which is the whole
point, since the question is which measurement would buy the most credibility.

Two outputs are probabilities and two are qualitative claims. For the
qualitative ones we report the *breaking factor* instead: how far the parameter
has to move before the paper's stated conclusion flips.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable

from . import ctmc
from . import params as P
from .threshold import (p_authority_assemblable, p_unauthorised_restoration,
                        quorum_rate)


@dataclass(frozen=True)
class Scenario:
    """A complete parameterisation, so a parameter can be varied in isolation."""

    holders: tuple[P.Holder, ...] = P.HOLDERS
    possession_h: float = P.POSSESSION_POOLED_H
    response_rate: float = P.RESPONSE_RATE
    write_off_h: float = P.WRITE_OFF_HORIZON_H
    p_modified: float = P.P_MODIFIED
    remediation_rate: float = P.REMEDIATION_RATE
    restoration_rate_h1: float = P.RESTORATION_RATE["H1"]
    #: Only B0 has a published extraction time. H1 is rated rather than priced
    #: (Section `sec:ap`), so there is no H1 extraction parameter to perturb and
    #: no elasticity of any result with respect to one.
    t_ext_b0_a2: float = P.TIME_TO_EXTRACTION_H["B0"]["A2"]

    def solve(self, t: int, variant: str = "H1", n: int | None = None):
        holders = self.holders[:n] if n else self.holders
        p_a = p_authority_assemblable(holders, t)
        lam_g = quorum_rate(holders, t, self.response_rate)
        T, A = ctmc.build_generator(
            mean_possession_h=self.possession_h,
            time_to_extraction_h=(self.t_ext_b0_a2 if variant == "B0"
                                  else float("inf")),
            restoration_rate=(self.restoration_rate_h1 if variant == "H1"
                              else P.RESTORATION_RATE["B0"]),
            p_modified_effective=self.p_modified if variant == "H1" else 0.0,
            quorum_gather_rate=lam_g,
            p_authority=p_a,
            remediation_rate=self.remediation_rate,
            write_off_horizon_h=self.write_off_h,
        )
        B, N = ctmc.solve(T, A)
        return B, N, p_a, holders


# --- the four outputs --------------------------------------------------------

def out_p_ext_b0(s: Scenario) -> float:
    """The phase-one figure the paper does publish: the baseline's exposure."""
    B, _, _, _ = s.solve(2, "B0", n=4)
    return float(B[0, 1])


def out_p_loss_given_custody(s: Scenario) -> float:
    """The commercial cost of the recovery design, at the recommended 2-of-4."""
    B, _, p_a, _ = s.solve(2, "H1", n=4)
    import numpy as np
    pi = np.zeros(5)
    pi[1], pi[2] = p_a, 1.0 - p_a
    return float((pi @ B)[2])


def out_e_tc(s: Scenario) -> float:
    """Restoration latency at 2-of-4."""
    import numpy as np
    B, N, p_a, _ = s.solve(2, "H1", n=4)
    t_ok = ctmc.conditional_mean_time(N, B, "OK")
    pi = np.zeros(5)
    pi[1], pi[2] = p_a, 1.0 - p_a
    mass = pi * B[:, 0]
    return float(np.nansum(mass * np.nan_to_num(t_ok)) / mass.sum())


def out_p_rest_given_custody(s: Scenario) -> float:
    """The recovery side of the same design point."""
    import numpy as np
    B, _, p_a, _ = s.solve(2, "H1", n=4)
    pi = np.zeros(5)
    pi[1], pi[2] = p_a, 1.0 - p_a
    return float((pi @ B)[0])


OUTPUTS: dict[str, Callable[[Scenario], float]] = {
    "P(ext) B0": out_p_ext_b0,
    "P(rest|c)": out_p_rest_given_custody,
    "P(loss|c)": out_p_loss_given_custody,
    "E[T_c]": out_e_tc,
}


# --- the parameters ----------------------------------------------------------

def _scale_holders(s: Scenario, field: str, factor: float) -> Scenario:
    cap = 0.999
    return replace(s, holders=tuple(
        replace(h, **{field: min(getattr(h, field) * factor, cap)})
        for h in s.holders))


PARAMETERS: dict[str, tuple[str, Callable[[Scenario, float], Scenario]]] = {
    "possession duration": (
        "ASAM", lambda s, f: replace(s, possession_h=s.possession_h * f)),
    "time to extraction (B0)": (
        "stipulated", lambda s, f: replace(s, t_ext_b0_a2=s.t_ext_b0_a2 * f)),
    "write-off horizon": (
        "stipulated", lambda s, f: replace(s, write_off_h=s.write_off_h * f)),
    "holder availability a_i": (
        "stipulated", lambda s, f: _scale_holders(s, "availability", f)),
    "holder compromise c_i": (
        "stipulated", lambda s, f: _scale_holders(s, "compromise", f)),
    "response rate rho": (
        "stipulated", lambda s, f: replace(s, response_rate=s.response_rate * f)),
    "P(modified)": (
        "stipulated", lambda s, f: replace(s, p_modified=min(s.p_modified * f, 0.999))),
    "remediation rate": (
        "stipulated",
        lambda s, f: replace(s, remediation_rate=s.remediation_rate * f)),
    "restoration rate (H1)": (
        "stipulated",
        lambda s, f: replace(s, restoration_rate_h1=s.restoration_rate_h1 * f)),
}


def elasticity(output: Callable[[Scenario], float],
               perturb: Callable[[Scenario, float], Scenario],
               base: Scenario = Scenario(), step: float = 0.10) -> float:
    """Symmetric log-derivative of ``output`` with respect to one parameter."""
    hi, lo = output(perturb(base, 1 + step)), output(perturb(base, 1 - step))
    if hi <= 0 or lo <= 0:
        return float("nan")
    return (math.log(hi) - math.log(lo)) / (math.log(1 + step) - math.log(1 - step))


def elasticity_table(step: float = 0.10) -> list[dict[str, object]]:
    """Elasticity of every output with respect to every parameter."""
    base = Scenario()
    rows = []
    for name, (source, perturb) in PARAMETERS.items():
        row: dict[str, object] = {"parameter": name, "source": source}
        for out_name, out_fn in OUTPUTS.items():
            row[out_name] = elasticity(out_fn, perturb, base, step)
        row["max |E|"] = max(abs(float(row[o])) for o in OUTPUTS
                             if not math.isnan(float(row[o])))
        rows.append(row)
    return sorted(rows, key=lambda r: -float(r["max |E|"]))


# --- how far can a parameter move before a conclusion flips? -----------------

def _finding1_holds(s: Scenario) -> bool:
    """Widening n dominates raising t, on all three post-custody axes."""
    import numpy as np

    def post(n: int, t: int) -> tuple[float, float, float]:
        B, N, p_a, _ = s.solve(t, "H1", n=n)
        pi = np.zeros(5)
        pi[1], pi[2] = p_a, 1.0 - p_a
        b = pi @ B
        t_ok = ctmc.conditional_mean_time(N, B, "OK")
        mass = pi * B[:, 0]
        return (float(b[0]), float(b[2]),
                float(np.nansum(mass * np.nan_to_num(t_ok)) / mass.sum()))

    a, b = post(3, 2), post(4, 2)
    return b[0] > a[0] and b[1] < a[1] and b[2] < a[2]


def _viable_band(s: Scenario) -> list[int]:
    import numpy as np
    band = []
    for t in range(1, len(s.holders) + 1):
        if p_unauthorised_restoration(s.holders, t) >= 1e-2:
            continue
        B, _, p_a, _ = s.solve(t, "H1")
        pi = np.zeros(5)
        pi[1], pi[2] = p_a, 1.0 - p_a
        if float((pi @ B)[2]) < 0.2:
            band.append(t)
    return band


CLAIMS: dict[str, Callable[[Scenario], bool]] = {
    "Finding 1 (widen n before raising t)": _finding1_holds,
    "Finding 2 (viable band is {2,3})": lambda s: _viable_band(s) == [2, 3],
    "B0 loses A_cpi in most incidents": lambda s: out_p_ext_b0(s) > 0.5,
}


def breaking_factors(factors: tuple[float, ...] = (
        0.02, 0.05, 0.1, 0.2, 0.33, 0.5, 0.67, 0.8, 1.25, 1.5, 2.0, 3.0, 5.0,
        10.0, 20.0, 50.0)) -> list[dict[str, object]]:
    """Smallest multiplicative perturbation of each parameter that breaks a claim.

    Reported as the factor furthest from 1 that the parameter can take while
    every claim still holds -- i.e. the width of the safe range. A claim that
    survives a 50x error in a parameter is not really resting on that parameter.
    """
    base = Scenario()
    rows = []
    for name, (source, perturb) in PARAMETERS.items():
        row: dict[str, object] = {"parameter": name, "source": source}
        for claim, holds in CLAIMS.items():
            broken_down = broken_up = None
            for f in sorted((x for x in factors if x < 1), reverse=True):
                if not holds(perturb(base, f)):
                    broken_down = f
                    break
            for f in sorted(x for x in factors if x > 1):
                if not holds(perturb(base, f)):
                    broken_up = f
                    break
            row[claim] = (broken_down, broken_up)
        rows.append(row)
    return rows


def format_report(step: float = 0.10) -> str:
    lines: list[str] = []
    lines.append("Elasticity  E = dln(output)/dln(parameter), "
                 f"symmetric over +/-{step:.0%}")
    lines.append("|E| >= 1 means a 10% error in the parameter is a >=10% error "
                 "in the result.")
    lines.append("")
    header = f"{'parameter':<26}{'source':<12}" + "".join(
        f"{o:>15}" for o in OUTPUTS)
    lines.append(header)
    lines.append("-" * len(header))
    for row in elasticity_table(step):
        lines.append(f"{row['parameter']:<26}{row['source']:<12}"
                     + "".join(f"{float(row[o]):>15.3f}" for o in OUTPUTS))

    lines.append("")
    lines.append("Robustness: multiplicative error a parameter can absorb "
                 "before a stated claim flips")
    lines.append("('-' = claim survives every factor tested, 0.02x to 50x)")
    lines.append("")
    for row in breaking_factors():
        broken = {c: v for c, v in row.items()
                  if c not in ("parameter", "source") and any(v)}
        if not broken:
            lines.append(f"  {row['parameter']:<26} robust to all tested claims")
            continue
        lines.append(f"  {row['parameter']:<26} ({row['source']})")
        for claim, (down, up) in broken.items():
            d = f"{down:g}x" if down else "-"
            u = f"{up:g}x" if up else "-"
            lines.append(f"      breaks {claim:<38} at {d} down / {u} up")
    return "\n".join(lines)
