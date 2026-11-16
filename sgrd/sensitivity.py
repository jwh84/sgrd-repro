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

import itertools
import math
from dataclasses import dataclass, replace
from typing import Callable

from . import ctmc
from . import params as P
from .threshold import (p_at_least, p_authority_assemblable, quorum_rate, p_unauthorised_restoration,
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
    pi = np.zeros(len(ctmc.TRANSIENT))
    pi[1], pi[2] = p_a, 1.0 - p_a
    return float((pi @ B)[2])


def out_e_tc(s: Scenario) -> float:
    """Restoration latency at 2-of-4."""
    import numpy as np
    B, N, p_a, _ = s.solve(2, "H1", n=4)
    t_ok = ctmc.conditional_mean_time(N, B, "OK")
    pi = np.zeros(len(ctmc.TRANSIENT))
    pi[1], pi[2] = p_a, 1.0 - p_a
    mass = pi * B[:, 0]
    return float(np.nansum(mass * np.nan_to_num(t_ok)) / mass.sum())


def out_p_rest_given_custody(s: Scenario) -> float:
    """The recovery side of the same design point."""
    import numpy as np
    B, _, p_a, _ = s.solve(2, "H1", n=4)
    pi = np.zeros(len(ctmc.TRANSIENT))
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
        pi = np.zeros(len(ctmc.TRANSIENT))
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
        pi = np.zeros(len(ctmc.TRANSIENT))
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


#: The named configurations of Table `tab:ctmc` that meet the acceptance test.
MARGIN_CONFIGS: tuple[tuple[str, int, tuple[str, ...]], ...] = (
    ("2-of-3", 2, ("Q1", "Q2", "Q3")),
    ("2-of-4", 2, ("Q1", "Q2", "Q3", "Q4")),
    ("3-of-4", 3, ("Q1", "Q2", "Q3", "Q4")),
    ("3-of-5", 3, ("Q1", "Q2", "Q3", "Q4", "Q5")),
)
#: P_u half of the acceptance test in Finding 2.
P_U_LIMIT = 1e-2
#: Per-holder values for the exhaustive grid: 7^5 = 16,807 combinations.
C_GRID = (0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)


def scaled_p_u(compromise: list[float], t: int, scale: float) -> float:
    """P_u with every c_i multiplied by ``scale``, holders independent."""
    return p_at_least([min(1.0, c * scale) for c in compromise], t)


def compromise_margin() -> list[dict[str, object]]:
    """How wrong can the stipulated c_i be before P_u fails the acceptance test?

    c_i is not an input to the chain, so only P_u moves. ``comonotone`` is the
    opposite extreme from independence: holders fall together, so at least t are
    compromised exactly when the t-th most exposed one is.
    """
    tags = sorted(h.tag for h in P.HOLDERS)
    grid = list(itertools.product(C_GRID, repeat=len(tags)))
    rows: list[dict[str, object]] = []
    for name, t, members in MARGIN_CONFIGS:
        c = [h.compromise for h in P.holder_set(*members)]
        lo, hi = 1e-3, 1e3
        for _ in range(100):
            mid = (lo * hi) ** 0.5
            if scaled_p_u(c, t, mid) < P_U_LIMIT:
                lo = mid
            else:
                hi = mid
        idx = [tags.index(m) for m in members]
        passed = sum(p_at_least([v[i] for i in idx], t) < P_U_LIMIT for v in grid)
        rows.append({
            "configuration": name,
            "t": t,
            "p_u": scaled_p_u(c, t, 1.0),
            "limit_scale": lo,
            "grid_pass_fraction": passed / len(grid),
            "comonotone_p_u": sorted(c, reverse=True)[t - 1],
        })
    return rows


def grid_ordering(better: str = "3-of-5", worse: str = "2-of-4") -> float:
    """Fraction of the C_GRID combinations in which ``better`` has the lower P_u."""
    conf = {name: (t, members) for name, t, members in MARGIN_CONFIGS}
    tags = sorted(h.tag for h in P.HOLDERS)
    grid = list(itertools.product(C_GRID, repeat=len(tags)))
    (t_b, m_b), (t_w, m_w) = conf[better], conf[worse]
    i_b, i_w = [tags.index(m) for m in m_b], [tags.index(m) for m in m_w]
    wins = sum(p_at_least([v[i] for i in i_b], t_b)
               < p_at_least([v[i] for i in i_w], t_w) for v in grid)
    return wins / len(grid)


#: Loss half of the acceptance test in Finding 2.
LOSS_LIMIT = 0.2
#: Per-holder values for the exhaustive availability grid.
A_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99)   # steps of 0.1


def loss_given_custody(name: str, t: int, availability: list[float],
                       members: tuple[str, ...]) -> float:
    """P(loss | c) with the given a_i. c_i does not enter the chain."""
    holders = tuple(replace(h, availability=min(1.0, a))
                    for h, a in zip(P.holder_set(*members), availability))
    return ctmc.evaluate(P.Configuration(name, t, holders, "H1")).p_loss_given_custody


def availability_margin() -> list[dict[str, object]]:
    """How far can the stipulated a_i fall before the loss test fails?

    The counterpart of :func:`compromise_margin`. a_i moves only the chain and c_i
    moves only P_u, so the two margins are independent and the acceptance region is
    their product.
    """
    rows: list[dict[str, object]] = []
    for name, t, members in MARGIN_CONFIGS:
        a = [h.availability for h in P.holder_set(*members)]
        lo, hi = 0.0, 1.0        # lo fails the test, hi passes
        for _ in range(60):
            mid = (lo + hi) / 2
            if loss_given_custody(name, t, [x * mid for x in a], members) < LOSS_LIMIT:
                hi = mid
            else:
                lo = mid
        single = {}
        for j, tag in enumerate(members):
            l2, h2 = 0.0, a[j]
            base = loss_given_custody(name, t, [0.0 if k == j else x for k, x in enumerate(a)],
                                      members) if t <= len(a) - 1 else 1.0
            if base < LOSS_LIMIT:
                single[tag] = 0.0   # the test holds even if this holder never answers
                continue
            for _ in range(50):
                mid = (l2 + h2) / 2
                v = list(a)
                v[j] = mid
                if loss_given_custody(name, t, v, members) < LOSS_LIMIT:
                    h2 = mid
                else:
                    l2 = mid
            single[tag] = h2
        grid = list(itertools.product(A_GRID, repeat=len(members)))
        passed = sum(loss_given_custody(name, t, list(v), members) < LOSS_LIMIT for v in grid)
        rows.append({
            "configuration": name,
            "t": t,
            "p_loss": loss_given_custody(name, t, a, members),
            "limit_scale": hi,
            "single_holder_floor": single,
            "grid_pass_fraction": passed / len(grid),
            "grid_size": len(grid),
        })
    return rows


#: Low / base / high for every other stipulated value, as hours or a probability.
#: The ends bracket what could plausibly happen; they are not estimates.
JOINT_RANGES: dict[str, tuple[float, float, float]] = {
    "response_h": (1.0, 1.0 / P.RESPONSE_RATE, 24.0),
    "restoration_h": (2.0, 1.0 / P.RESTORATION_RATE["H1"], 72.0),
    "remediation_h": (12.0, 1.0 / P.REMEDIATION_RATE, 336.0),
    "write_off_h": (168.0, P.WRITE_OFF_HORIZON_H, 8760.0),
    "p_modified": (0.0, P.P_MODIFIED, 1.0),
}
#: The levels actually enumerated. p_m is a probability, so it steps by 0.1 like
#: a_i; the durations have no natural step and stay at low / base / high.
JOINT_LEVELS: dict[str, tuple[float, ...]] = {
    **JOINT_RANGES,
    "p_modified": tuple(i / 10 for i in range(11)),
}


def closed_form_loss(p_a, gather_at_unit_rate, response_h, restoration_h,
                     remediation_h, write_off_h, p_modified):
    """P(loss | c) of the chain, in closed form, so the grid can be vectorised.

    From Z_u the only exit is write-off. From Z_r the episode must beat the
    write-off hazard at each stage it passes through: gathering, restoring and,
    with probability p_modified, remediating -- after which the quorum is
    gathered again for a token over the new measured state and the stages are
    walked a second time (ZR2, RS2 in :mod:`sgrd.ctmc`).
    """
    w = 1.0 / write_off_h
    gather = gather_at_unit_rate / response_h   # the rate scales with rho
    mu = 1.0 / restoration_h
    rem = 1.0 / remediation_h
    after_remediation = rem / (rem + w) * gather / (gather + w) * mu / (mu + w)
    ok_from_rs = mu * ((1.0 - p_modified) + p_modified * after_remediation) / (mu + w)
    return 1.0 - p_a * gather / (gather + w) * ok_from_rs


def joint_range() -> dict[str, object]:
    """Every a_i level with every low/base/high combination of the other values.

    a_i and the five others move only the loss; c_i moves only P_u. The two parts
    are computed apart and combine as a product, so nothing is sampled.
    """
    import numpy as np

    names = list(JOINT_RANGES)
    combos = list(itertools.product(*(JOINT_LEVELS[k] for k in names)))
    arr = {k: np.array([c[i] for c in combos]) for i, k in enumerate(names)}
    base = {k: JOINT_RANGES[k][1] for k in names}
    margins = {r["configuration"]: r for r in compromise_margin()}

    def grid_values(t: int, members: tuple[str, ...]):
        p_a, g = [], []
        for v in itertools.product(A_GRID, repeat=len(members)):
            hs = [replace(h, availability=a) for h, a in zip(P.holder_set(*members), v)]
            p = p_authority_assemblable(hs, t)
            p_a.append(p)
            g.append(quorum_rate(hs, t, 1.0))
        return np.array(p_a), np.array(g)

    out: list[dict[str, object]] = []
    losses: dict[str, np.ndarray] = {}
    for name, t, members in MARGIN_CONFIGS:
        hs = P.holder_set(*members)
        pa0 = p_authority_assemblable(hs, t)
        g0 = quorum_rate(hs, t, 1.0)
        at_base_a = closed_form_loss(pa0, g0, **arr)
        pa, g = grid_values(t, members)
        # shape: (a combinations, other combinations)
        full = closed_form_loss(pa[:, None], g[:, None], **arr)
        losses[name] = full
        worst = int(np.argmax(at_base_a))
        single = {}
        for k in names:
            lo = closed_form_loss(pa0, g0, **{**base, k: JOINT_RANGES[k][0]})
            hi = closed_form_loss(pa0, g0, **{**base, k: JOINT_RANGES[k][2]})
            single[k] = (float(lo), float(hi))
        a_pass = float((full < LOSS_LIMIT).mean())
        c_pass = margins[name]["grid_pass_fraction"]
        out.append({
            "configuration": name,
            "t": t,
            "base_loss": float(closed_form_loss(pa0, g0, **base)),
            "combinations": int(full.size),
            "others_pass_at_base_a": float((at_base_a < LOSS_LIMIT).mean()),
            "worst_others_loss_at_base_a": float(at_base_a[worst]),
            "worst_others": dict(zip(names, combos[worst])),
            "loss_range": (float(full.min()), float(full.max())),
            "all_values_pass": a_pass,
            "with_c_pass": a_pass * float(c_pass),
            "single": single,
        })

    # Finding 1: does adding Q4 to 2-of-3 ever raise the loss? Q4's a is free.
    three, four = losses["2-of-3"], losses["2-of-4"]
    n = len(A_GRID)
    expanded = np.repeat(three, n, axis=0)   # each a over Q1..Q3, for each a_Q4
    return {
        "rows": out,
        "finding1_loss_not_worse": float((four <= expanded + 1e-12).mean()),
        "others_combinations": len(combos),
    }


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

    lines.append("")
    lines.append(f"Compromise margin: common factor on every c_i before P_u >= "
                 f"{P_U_LIMIT:g} (grid: {len(C_GRID)}^{len(P.HOLDERS)} per-holder values)")
    lines.append("")
    for row in compromise_margin():
        lines.append(f"  {row['configuration']:<8} P_u={row['p_u']:.2e}  "
                     f"limit x{row['limit_scale']:.2f}  "
                     f"grid pass {row['grid_pass_fraction']:.1%}  "
                     f"comonotone P_u={row['comonotone_p_u']:.2f}")
    lines.append(f"  3-of-5 below 2-of-4 in {grid_ordering():.1%} of the grid")

    lines.append("")
    lines.append(f"Availability margin: common factor on every a_i below which "
                 f"P(loss|c) >= {LOSS_LIMIT:g}")
    lines.append("")
    for row in availability_margin():
        lines.append(f"  {row['configuration']:<8} loss={row['p_loss']:.3f}  "
                     f"limit x{row['limit_scale']:.2f}  "
                     f"grid pass {row['grid_pass_fraction']:.1%} of {row['grid_size']}  "
                     f"floors {row['single_holder_floor']}")

    joint = joint_range()
    lines.append("")
    lines.append(f"Joint ranges: every a_i level x {joint['others_combinations']} "
                 "low/base/high combinations of the other values")
    lines.append("")
    for row in joint["rows"]:
        lines.append(f"  {row['configuration']:<8} base a_i: {row['others_pass_at_base_a']:.1%} pass, "
                     f"worst loss {row['worst_others_loss_at_base_a']:.3f}; "
                     f"all values: {row['all_values_pass']:.1%}; with c_i: {row['with_c_pass']:.1%}")
    lines.append(f"  2-of-4 no worse than 2-of-3 in {joint['finding1_loss_not_worse']:.1%}")
    return "\n".join(lines)
