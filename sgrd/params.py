"""Parameterisation of the SGRD analysis (paper Table `tab:params`).

Every value that enters the quantitative model lives here, tagged with its
provenance so that the `source` column of the paper's parameter table can be
audited mechanically. Values marked ``STIPULATED`` are authors' assumptions and
are the ones a reviewer should vary; values marked ``ASAM`` are derived from the
public NGA incident database (see :mod:`sgrd.asam`); values marked ``ANCHORED``
are tied to a published evaluation claim rather than measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Source(str, Enum):
    STIPULATED = "stipulated"
    ASAM = "ASAM"
    ANCHORED = "anchored"


@dataclass(frozen=True)
class Holder:
    """A holder of one threshold share of the recovery key ``K_r``.

    Tagged ``Q1..Qn`` rather than ``H1..Hn``: ``H1`` is also the name of the
    hardened controller variant, and the two met in expressions such as
    ``Configuration(name, t, holder_set("H1", ...), "H1")`` where only position
    distinguished a holder from a design.
    """

    tag: str
    role: str
    # a_i: probability that this holder takes part in the episode at all --
    # reachable, authorised and willing. Not a deadline: latency given
    # participation is exponential at RESPONSE_RATE, drawn separately.
    availability: float
    compromise: float  # c_i: probability of being coerced or bribed

    def __post_init__(self) -> None:
        for name, v in (("availability", self.availability),
                        ("compromise", self.compromise)):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{self.tag}.{name} must be a probability, got {v}")


# --- Table tab:params, holder block -----------------------------------------

HOLDERS: tuple[Holder, ...] = (
    Holder("Q1", "ROC watch, 24/7", 0.99, 0.05),
    Holder("Q2", "classification society", 0.70, 0.02),
    Holder("Q3", "shipowner escrow", 0.60, 0.03),
    Holder("Q4", "commercial 24/7 escrow", 0.98, 0.01),
    Holder("Q5", "flag administration", 0.50, 0.01),
)

HOLDERS_BY_TAG = {h.tag: h for h in HOLDERS}


def holder_set(*tags: str) -> tuple[Holder, ...]:
    """Select holders by tag, preserving the given order."""
    try:
        return tuple(HOLDERS_BY_TAG[t] for t in tags)
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown holder tag {exc.args[0]!r}; "
                       f"known: {sorted(HOLDERS_BY_TAG)}") from exc


# --- Table tab:params, scalar block -----------------------------------------

#: Mean possession duration, pooled over the audited ASAM sample (hours).
POSSESSION_POOLED_H = 175.5

#: Mean possession duration by NAVAREA stratum (hours). Used for the regional
#: sensitivity analysis of Section `sec:results`.
POSSESSION_STRATA_H: dict[str, float] = {
    "NAVAREA II (Gulf of Guinea)": 86.4,
    "Other NAVAREAs (VIII, IX, XI)": 324.0,
}

#: Per-holder response rate rho, in 1/h.
RESPONSE_RATE = 0.25

#: Commercial write-off horizon (hours). Modelled as a competing exponential
#: risk out of the post-custody states only (Z_r, Z_u, RS, RM). It does not
#: compete from Z_p: declaring a constructive total loss is an accounting event
#: and does not end the adversary's physical access.
WRITE_OFF_HORIZON_H = 720.0

#: P(the adversary modified software/models/configuration during possession).
P_MODIFIED = 0.60

#: Off-board remediation rate after a failed R1 attestation, in 1/h.
REMEDIATION_RATE = 1.0 / 24.0

#: Staged-restoration completion rate, in 1/h, per controller variant.
#: B0 restores in a single step; H1 walks R0->R1->R2->R3.
RESTORATION_RATE: dict[str, float] = {"B0": 1.0, "H1": 1.0 / 6.0}

#: Variants that can mint restoration authority on the platform. For these the
#: quorum is not the only route back, so P(D_c >= t) does not bound unauthorised
#: restoration and the chain reports 1 instead. This is the chain's counterpart
#: of the P1 failure in `sgrd.modelcheck`: the same weakness, priced rather than
#: enumerated.
ONBOARD_RECOVERY_AUTHORITY: frozenset[str] = frozenset({"B0"})

#: Mean time to extraction of A_cpi once the adversary has the hardware (hours),
#: by adversary class. B0 only.
#:
#: B0 retains sealed key material on board, so extraction is a key-recovery
#: problem whose difficulty needs no evidence to state. H1 is deliberately absent
#: here: Section `sec:ap` rates it rather than pricing it, because its resistance
#: is conditional on zeroisation evidence and an attack-potential band is an
#: ordinal statement of the resources a path requires, not a mean time. The H1
#: figures the chain would need are therefore not published, and no result in the
#: paper depends on them.
TIME_TO_EXTRACTION_H: dict[str, dict[str, float]] = {
    "B0": {"A2": 72.0, "A3": 24.0},
}

PARAMETER_SOURCES: dict[str, Source] = {
    "holder participation probabilities a_i": Source.STIPULATED,
    "holder compromise probabilities c_i": Source.STIPULATED,
    "mean possession duration, pooled": Source.ASAM,
    "mean possession duration, regional strata": Source.ASAM,
    "per-holder response rate rho": Source.STIPULATED,
    "commercial write-off horizon": Source.STIPULATED,
    "P(modified during possession)": Source.STIPULATED,
    "off-board remediation rate": Source.STIPULATED,
    "restoration rate, B0 / H1": Source.STIPULATED,
    "time to extraction, B0": Source.STIPULATED,
}


@dataclass(frozen=True)
class Configuration:
    """A named (t, holder-set, controller-variant) design point."""

    name: str
    t: int
    holders: tuple[Holder, ...]
    variant: str  # a key of RESTORATION_RATE: "B0" or "H1"

    def __post_init__(self) -> None:
        if self.variant not in RESTORATION_RATE:
            raise ValueError(
                f"variant must be one of {sorted(RESTORATION_RATE)}, "
                f"got {self.variant!r}")
        if not 1 <= self.t <= len(self.holders):
            raise ValueError(
                f"{self.name}: threshold t={self.t} outside 1..n={len(self.holders)}")

    @property
    def n(self) -> int:
        return len(self.holders)


#: The named configurations of paper Table `tab:ctmc`.
def named_configurations() -> list[Configuration]:
    roc = holder_set("Q1")
    rco = holder_set("Q1", "Q2", "Q3")
    rcoe = holder_set("Q1", "Q2", "Q3", "Q4")
    rcoef = holder_set("Q1", "Q2", "Q3", "Q4", "Q5")
    # B0 appears once. Its restoration authority is on the platform, so it has
    # no threshold and its post-custody outcome is the same whatever holder set
    # is nominally attached; listing it per configuration implied a dependence
    # the design does not have.
    return [
        Configuration("local secret (no quorum)", 1, roc, "B0"),
        Configuration("1-of-1 (ROC only)", 1, roc, "H1"),
        Configuration("2-of-3 (ROC, class, owner)", 2, rco, "H1"),
        Configuration("2-of-4 (+ 24/7 escrow)", 2, rcoe, "H1"),
        Configuration("3-of-4 (+ 24/7 escrow)", 3, rcoe, "H1"),
        Configuration("3-of-5 (+ flag)", 3, rcoef, "H1"),
    ]


#: Values printed in the paper, used by :mod:`sgrd.report` to check the
#: reimplementation against the published article rather than against itself.
PUBLISHED = {
    # Table tab:ctmc, adversary A2, pooled possession regime.
    # key -> (P_ext, P_rest_given_custody, P_loss_given_custody, E_Tc, P_u)
    "ctmc": {
        ("local secret (no quorum)", "B0"): (0.709, 0.999, 0.001, 1.0, 1.0),
        ("1-of-1 (ROC only)", "H1"): (None, 0.958, 0.043, 23.7, 5.0e-2),
        ("2-of-3 (ROC, class, owner)", "H1"): (None, 0.846, 0.154, 24.4, 3.0e-3),
        ("2-of-4 (+ 24/7 escrow)", "H1"): (None, 0.965, 0.035, 23.0, 4.0e-3),
        ("3-of-4 (+ 24/7 escrow)", "H1"): (None, 0.836, 0.164, 25.6, 6.0e-5),
        ("3-of-5 (+ flag)", "H1"): (None, 0.899, 0.101, 24.7, 9.9e-5),
    },
    # Section sec:results, phase one. (variant, adversary) -> P_ext, pooled.
    "extraction_pooled": {
        ("B0", "A2"): 0.709,
        ("B0", "A3"): 0.880,
    },
    # Fig. fig:regimes: H1 residual extraction risk under A2 by stratum mean.
    "extraction_regimes_b0_a2": {86.4: 0.545, 324.0: 0.818},
    # Finding 2's valuation. C_u / C_hull at which 2-of-4 and the named
    # configuration cost the same, under P_u * C_u + P(loss|c) * C_hull. C_u is
    # the cost of an *unauthorised restoration* -- at least the autonomy stack,
    # and at most what a hostile party can do with a laden ship under power.
    "cost_ratio_vs_2of4": {"3-of-5 (+ flag)": 16.8, "3-of-4 (+ 24/7 escrow)": 32.9},
    # Finding 3: unconditional permanent-loss probability at 2-of-4, A2, pooled.
    "unconditional_loss_b0": {"B0": 0.000},
    # Finding 2: the t sweep over the full five-holder set.
    "sweep_five_holders": {
        1: {"P_u": 0.115},
        4: {"P_available": 0.637, "P_loss_given_custody": 0.387},
        5: {"P_available": 0.204, "P_loss_given_custody": 0.804},
    },
    # Table tab:asam.
    "asam": {
        "Pooled": {"n": 16, "median": 120.0, "mean": 175.5, "p90": 444.0, "max": 648.0},
        "NAVAREA II (Gulf of Guinea)": {"n": 10, "median": 84.0, "mean": 86.4,
                                        "p90": 146.4, "max": 168.0},
        "Other NAVAREAs (VIII, IX, XI)": {"n": 6, "median": 228.0, "mean": 324.0,
                                          "p90": 624.0, "max": 648.0},
    },
    # Table tab:mc: witness counts and reachable-relation sizes. B1--B3 are the
    # intermediate designs: each holds all of SGRD but one contribution, so the
    # column that fails names what that contribution discharges.
    "modelcheck": {
        "B0": {"states": 722, "transitions": 3116, "satisfied": 1,
               "witnesses": {"P1": 96, "P2": 176, "P3": 128, "P4": 136,
                             "P5": 272, "P6": 272, "P7": 76, "P8": 136,
                             "P9": 0, "P10": 96, "P11": 1356, "P12": 416}},
        "B1": {"states": 514, "transitions": 2280, "satisfied": 11,
               "witnesses": {**{f"P{i}": 0 for i in range(1, 13)}, "P10": 4}},
        "B2": {"states": 234, "transitions": 916, "satisfied": 9,
               "witnesses": {**{f"P{i}": 0 for i in range(1, 13)},
                             "P5": 16, "P6": 16, "P8": 8}},
        "B3": {"states": 266, "transitions": 1104, "satisfied": 11,
               "witnesses": {**{f"P{i}": 0 for i in range(1, 13)}, "P4": 4}},
        "H1": {"states": 346, "transitions": 1430, "satisfied": 12,
               "witnesses": {f"P{i}": 0 for i in range(1, 13)}},
    },
    # Section sec:results: which flag actually discharges which property.
    # Seven are load-bearing in the model; three are design requirements for
    # reasons the model does not represent.
    "ablation": {
        "erase_keys_on_denial": ("P7",),
        "onboard_recovery_secret": (),
        "monotonic_counter": ("P9",),
        "staged": ("P5", "P6", "P8"),
        "gate_custody": ("P3", "P12"),
        "gate_safe": ("P4",),
        "gate_attested": (),
        "gate_unused": (),
        "require_threshold": ("P10",),
        "nav_kernel_outside_denial": ("P11",),
    },
}
