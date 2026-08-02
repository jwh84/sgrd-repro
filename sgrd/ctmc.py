"""The absorbing continuous-time Markov chain of Section `sec:method`.

Time starts at the instant denial fires. Transient states:

    Z_p   denied, adversary still in physical possession
    Z_r   custody regained, restoration authority assemblable
    Z_u   custody regained, authority NOT assemblable (fewer than t holders)
    RS    staged restoration in progress
    RM    off-board remediation after a failed R1 attestation

Absorbing states:

    OK    capability restored
    EXT   A_cpi extracted by the adversary
    BRK   permanent commercial loss (constructive total loss)

Three structural commitments, all of them load-bearing:

1. **Extraction fires only from Z_p.** The adversary needs the hardware. Once
   custody is back, ``A_cpi`` is no longer at risk, so ``P(EXT)`` is decided
   entirely by the race in Z_p and is *independent of t*. This is what lets the
   paper report the two phases separately.

2. **Restoration begins only after custody is regained.** That is not an extra
   assumption bolted onto the chain -- it is the ``NOT Possessed`` conjunct of
   the R0 gate in Section `sec:staged`, carried over.

3. **The write-off horizon competes only after custody is regained.** Z_r, Z_u,
   RS and RM each carry an exponential hazard ``1/720 h`` into BRK; Z_p does
   not. A constructive total loss is an accounting event and does not end the
   adversary's physical access, so it cannot truncate an extraction in
   progress.

Only B0 is priced for extraction. Its weakness is structural -- sealed key
material sits on the platform -- so a mean time to extraction can be stated
without any evaluation evidence. H1's resistance is conditional on evidence that
the module's zeroisation is complete, and Section `sec:ap` therefore rates it in
an attack-potential band rather than pricing it; a band is an ordinal statement
of the resources a path requires, not a mean time, and collapsing one into the
other would manufacture a figure the evidence does not support. Every quantity
conditional on custody is still computed for H1, because those are entered at
Z_r/Z_u and never read Z_p's row.

The remaining B0/H1 difference inside the chain is whether a modification made
during possession is caught. B0 has no post-possession attestation, so it
restores a tampered configuration without noticing -- modelled as
``p_modified_effective = 0``, i.e. B0 never routes to RM. That is a *flattering*
modelling choice for B0: it counts a silently restored tampered stack as a
success, which is why Finding 3 matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import params as P
from .params import Configuration
from .threshold import p_authority_assemblable, p_unauthorised_restoration, quorum_rate

# RS (restoring), not RC: RC is the recovery capsule of Section `sec:design`.
TRANSIENT = ("Z_p", "Z_r", "Z_u", "RS", "RM")
ABSORBING = ("OK", "EXT", "BRK")

_TI = {s: i for i, s in enumerate(TRANSIENT)}
_AI = {s: i for i, s in enumerate(ABSORBING)}


@dataclass(frozen=True)
class Outcome:
    """Absorption probabilities and timings for one design point."""

    #: ``None`` for a variant the paper rates rather than prices: see
    #: :func:`evaluate`. Everything conditional on custody is still defined,
    #: because those quantities are entered at Z_r/Z_u and never read Z_p's row.
    p_extracted: float | None
    p_restored_given_custody: float
    p_loss_given_custody: float
    mean_time_to_restore_given_custody: float
    p_unauthorised: float
    # Unconditional absorption from the instant of denial; ``None`` when the
    # variant is unpriced, because an unconditional figure would silently carry
    # an extraction probability the paper does not claim.
    p_restored_uncond: float | None
    p_loss_uncond: float | None

    def as_row(self) -> dict[str, float]:
        return {
            "P(ext)": self.p_extracted,
            "P(rest|c)": self.p_restored_given_custody,
            "P(loss|c)": self.p_loss_given_custody,
            "E[T_c] (h)": self.mean_time_to_restore_given_custody,
            "P_u": self.p_unauthorised,
            "P(rest) uncond": self.p_restored_uncond,
            "P(loss) uncond": self.p_loss_uncond,
        }


def build_generator(
    *,
    mean_possession_h: float,
    time_to_extraction_h: float,
    restoration_rate: float,
    p_modified_effective: float,
    quorum_gather_rate: float,
    p_authority: float,
    remediation_rate: float = P.REMEDIATION_RATE,
    write_off_horizon_h: float = P.WRITE_OFF_HORIZON_H,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(T, A)``: the transient-to-transient and transient-to-absorbing
    blocks of the generator, ordered by :data:`TRANSIENT` / :data:`ABSORBING`.

    ``T`` carries the negative total exit rates on its diagonal, so ``-T`` is a
    non-singular M-matrix and ``(-T)^-1`` is the fundamental matrix.
    """
    if mean_possession_h <= 0 or time_to_extraction_h <= 0:
        raise ValueError("durations must be positive")
    # An infinite extraction time means "unpriced": no EXT edge at all.
    if not 0.0 <= p_modified_effective <= 1.0:
        raise ValueError("p_modified_effective must be a probability")
    if not 0.0 <= p_authority <= 1.0:
        raise ValueError("p_authority must be a probability")

    lam_ext = 1.0 / time_to_extraction_h
    lam_cust = 1.0 / mean_possession_h
    w = 1.0 / write_off_horizon_h

    nt, na = len(TRANSIENT), len(ABSORBING)
    T = np.zeros((nt, nt))
    A = np.zeros((nt, na))

    def t_edge(src: str, dst: str, rate: float) -> None:
        T[_TI[src], _TI[dst]] += rate

    def a_edge(src: str, dst: str, rate: float) -> None:
        A[_TI[src], _AI[dst]] += rate

    # Z_p: the adversary has the hardware. Two competing hazards.
    #
    # The write-off horizon does NOT compete here. A constructive total loss is
    # an accounting event: declaring it does not end the adversary's physical
    # possession, and does not stop the extraction that is under way. Treating
    # it as an exit from Z_p would let a bookkeeping decision truncate an attack,
    # which suppresses P(EXT) for both variants and by different factors. The
    # ASAM durations that set ``lam_cust`` are seizure-to-release intervals and
    # are therefore already conditioned on release, so leaving Z_p by regaining
    # custody is the modelled end of possession.
    a_edge("Z_p", "EXT", lam_ext)
    t_edge("Z_p", "Z_r", lam_cust * p_authority)
    t_edge("Z_p", "Z_u", lam_cust * (1.0 - p_authority))

    # Z_r: custody back, quorum reachable. Race the gather against write-off.
    t_edge("Z_r", "RS", quorum_gather_rate)
    a_edge("Z_r", "BRK", w)

    # Z_u: custody back, no quorum. The hull is recovered and then written off
    # because nobody can authorise restoration. This is the cost of raising t.
    a_edge("Z_u", "BRK", w)

    # RS: staged restoration. H1 routes a modified configuration to remediation
    # at the R1 gate; B0 has no such gate and restores it silently.
    a_edge("RS", "OK", restoration_rate * (1.0 - p_modified_effective))
    t_edge("RS", "RM", restoration_rate * p_modified_effective)
    a_edge("RS", "BRK", w)

    # RM: off-board remediation, then straight to capability. The configuration
    # is clean on return, so there is no second trip through RS.
    a_edge("RM", "OK", remediation_rate)
    a_edge("RM", "BRK", w)

    # Diagonal = -(total exit rate).
    exits = T.sum(axis=1) + A.sum(axis=1)
    np.fill_diagonal(T, T.diagonal() - exits)
    return T, A


def solve(T: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Absorption probabilities ``B`` and expected occupancy ``N``.

    ``N = (-T)^-1`` gives ``N[i, k]`` = expected total time spent in transient
    state ``k`` before absorption, starting from ``i``. ``B = N A`` gives the
    absorption probabilities.
    """
    N = np.linalg.solve(-T, np.eye(T.shape[0]))
    return N @ A, N


def conditional_mean_time(N: np.ndarray, B: np.ndarray, absorbing: str) -> np.ndarray:
    """E[time to absorption | absorbed in ``absorbing``], per starting state.

    Uses ``E[T * 1{absorb in j}] = sum_k N[i,k] * B[k,j]``: the sojourn time in
    each visit to ``k`` is independent of where the chain goes next, so the
    expected time in ``k`` and the probability of eventually being absorbed in
    ``j`` from ``k`` multiply. Dividing by ``B[:, j]`` conditions.
    """
    j = _AI[absorbing]
    numer = N @ B[:, j]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(B[:, j] > 0, numer / B[:, j], np.nan)
    return out


#: A local-secret baseline does not gather a quorum. Modelled as a rate fast
#: enough that the gather stage contributes nothing measurable, rather than by
#: deleting the state, so that both variants traverse the same chain.
_INSTANT_GATHER = 1e6


def evaluate(
    config: Configuration,
    *,
    adversary: str = "A2",
    mean_possession_h: float = P.POSSESSION_POOLED_H,
) -> Outcome:
    """Solve the chain for one named configuration."""
    # A variant that mints authority on the platform has no quorum to wait for
    # and none to assemble, so none of the three quorum quantities describes it:
    # authority is always available to whoever holds the hull (p_a = 1), the
    # gather stage is not traversed, and unauthorised restoration is not bounded
    # by P(D_c >= t) but is free. Computing them from a holder set anyway was
    # the modelling error this branch removes -- it made the baseline's
    # post-custody outcome vary with a threshold the baseline does not have.
    local = config.variant in P.ONBOARD_RECOVERY_AUTHORITY
    if local:
        p_a, p_u = 1.0, 1.0
        lam_g = _INSTANT_GATHER
    else:
        p_a = p_authority_assemblable(config.holders, config.t)
        p_u = p_unauthorised_restoration(config.holders, config.t)
        lam_g = quorum_rate(config.holders, config.t, P.RESPONSE_RATE)

    # A variant with no published extraction time is rated, not priced: Section
    # `sec:ap` places its remaining path in an attack-potential band, and a band
    # is an ordinal statement of the resources a path requires rather than a mean
    # time. Modelling it with no extraction edge keeps every post-custody
    # quantity exactly as it would be, and the absorption figures that would need
    # an extraction rate are returned as None rather than invented.
    priced = config.variant in P.TIME_TO_EXTRACTION_H
    t_ext = (P.TIME_TO_EXTRACTION_H[config.variant][adversary] if priced
             else float("inf"))

    T, A = build_generator(
        mean_possession_h=mean_possession_h,
        time_to_extraction_h=t_ext,
        restoration_rate=P.RESTORATION_RATE[config.variant],
        # No post-possession attestation in B0: nothing routes to remediation.
        p_modified_effective=P.P_MODIFIED if config.variant == "H1" else 0.0,
        quorum_gather_rate=lam_g,
        p_authority=p_a,
    )
    B, N = solve(T, A)
    t_ok = conditional_mean_time(N, B, "OK")

    # Post-custody view: enter at Z_r with probability p_a, else Z_u. This is
    # the conditional distribution of the chain the instant custody is regained,
    # and it is what the "| c" columns of Table tab:ctmc report.
    pi_c = np.zeros(len(TRANSIENT))
    pi_c[_TI["Z_r"]] = p_a
    pi_c[_TI["Z_u"]] = 1.0 - p_a
    b_c = pi_c @ B
    # Mix the conditional times by the mass each entry point contributes to OK.
    mass_ok = pi_c * B[:, _AI["OK"]]
    e_tc = float(np.nansum(mass_ok * np.nan_to_num(t_ok)) / mass_ok.sum())

    zp = _TI["Z_p"]
    return Outcome(
        p_extracted=float(B[zp, _AI["EXT"]]) if priced else None,
        p_restored_given_custody=float(b_c[_AI["OK"]]),
        p_loss_given_custody=float(b_c[_AI["BRK"]]),
        mean_time_to_restore_given_custody=e_tc,
        p_unauthorised=p_u,
        p_restored_uncond=float(B[zp, _AI["OK"]]) if priced else None,
        p_loss_uncond=float(B[zp, _AI["BRK"]]) if priced else None,
    )


def extraction_probability(
    variant: str, adversary: str, mean_possession_h: float
) -> float:
    """P(EXT) alone -- the phase-one result, independent of t and of the quorum.

    Closed form: a two-way race out of Z_p between extraction and the regaining
    of custody. The write-off horizon does not appear, because declaring a
    constructive total loss does not end the adversary's physical access
    (:func:`build_generator`).
    """
    if variant not in P.TIME_TO_EXTRACTION_H:
        raise KeyError(
            f"{variant} is rated, not priced: Section `sec:ap` places its "
            f"remaining path in an attack-potential band and the paper states "
            f"no mean time to extraction for it")
    if variant not in P.TIME_TO_EXTRACTION_H:
        raise KeyError(
            f"{variant} is rated, not priced: Section `sec:ap` places its "
            f"remaining path in an attack-potential band and the paper states "
            f"no mean time to extraction for it")
    lam_ext = 1.0 / P.TIME_TO_EXTRACTION_H[variant][adversary]
    lam_cust = 1.0 / mean_possession_h
    return lam_ext / (lam_ext + lam_cust)


def extraction_probability_erlang(
    variant: str, adversary: str, mean_possession_h: float, k: int = 1
) -> float:
    """P(EXT) when extraction is Erlang-``k`` with the same mean (Section
    `sec:disc`).

    The exponential of :func:`extraction_probability` gives extraction a
    constant hazard, so it admits completion arbitrarily early -- which is what
    an AVA_VAN band asserts does not happen. A physical extraction is a sequence
    of stages and has an *increasing* hazard. Erlang-k stays inside the CTMC:
    split ``Z_p`` into ``k`` stages each leaving at rate ``k * lam_ext``, so the
    mean is preserved and only the shape changes.

    Each stage races the same competing hazard ``mu`` (custody regained, or
    write-off), and ``mu`` is memoryless, so the k races are independent and

        P(EXT) = [ k*lam / (k*lam + mu) ]^k .

    ``k = 1`` recovers :func:`extraction_probability` exactly. Larger ``k``
    leaves the baseline almost unchanged and drives the hardened figure down,
    which is why the paper retains ``k = 1`` as the conservative case.
    """
    if k < 1:
        raise ValueError("k must be a positive number of stages")
    if variant not in P.TIME_TO_EXTRACTION_H:
        raise KeyError(f"{variant} is rated, not priced (see "
                       f"extraction_probability)")
    lam = 1.0 / P.TIME_TO_EXTRACTION_H[variant][adversary]
    mu = 1.0 / mean_possession_h
    return (k * lam / (k * lam + mu)) ** k
