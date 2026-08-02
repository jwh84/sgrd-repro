"""Threshold-authority arithmetic over heterogeneous holders.

The paper computes two quantities exactly "by dynamic programming over
heterogeneous holders":

* ``P(D_a >= t)`` -- the probability that at least ``t`` of the ``n`` holders are
  available within the response window, i.e. that legitimate restoration
  authority is assemblable at all;
* ``P(D_c >= t)`` -- the probability that at least ``t`` holders are compromised,
  i.e. that the adversary assembles authority (written ``P_u`` in Table
  `tab:ctmc`).

Both are Poisson-binomial tail probabilities. The DP below is the standard
convolution recurrence; it is exact in the sense that it performs no sampling,
though it is of course still floating point.

It also supplies the mean time to gather a quorum. Conditional on ``k >= t``
holders being available and each responding independently at rate ``rho``, the
wait for the ``j``-th response is exponential with rate ``(k - j) * rho``, so

    E[gather | k] = sum_{j=0}^{t-1} 1 / ((k - j) * rho)

which is the expression in Section `sec:method`. The unconditional mean used by
the Markov chain averages this over the availability distribution, conditioned
on a quorum existing.
"""

from __future__ import annotations

from collections.abc import Sequence

from .params import Holder


def poisson_binomial_pmf(probs: Sequence[float]) -> list[float]:
    """Exact PMF of the number of successes among independent Bernoulli trials.

    Returns a list of length ``len(probs) + 1`` whose ``k``-th entry is the
    probability of exactly ``k`` successes. Runs in O(n^2) by convolving one
    trial at a time.
    """
    pmf = [1.0]
    for p in probs:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability out of range: {p}")
        prev, pmf = pmf, [0.0] * (len(pmf) + 1)
        for k, mass in enumerate(prev):
            if mass == 0.0:
                continue
            pmf[k] += mass * (1.0 - p)
            pmf[k + 1] += mass * p
    return pmf


def p_at_least(probs: Sequence[float], t: int) -> float:
    """P(at least ``t`` successes). Summed from the tail to limit cancellation."""
    if t <= 0:
        return 1.0
    pmf = poisson_binomial_pmf(probs)
    if t > len(probs):
        return 0.0
    return sum(pmf[t:])


def p_authority_assemblable(holders: Sequence[Holder], t: int) -> float:
    """P(D_a >= t): legitimate authority can be gathered."""
    return p_at_least([h.availability for h in holders], t)


def p_unauthorised_restoration(holders: Sequence[Holder], t: int) -> float:
    """P(D_c >= t), written ``P_u`` in the paper: the adversary reaches quorum.

    This is the quantity Requirement `req:deny` bounds against adversary class
    A4 acting alone and class A5 acting with possession.
    """
    return p_at_least([h.compromise for h in holders], t)


def cost_ratio_indifference(p_u_from: float, loss_from: float,
                            p_u_to: float, loss_to: float) -> float:
    """Cost ratio ``C_u / C_hull`` at which two configurations cost the same.

    ``P_u`` is reported by the paper but is not an input to the chain, so the
    unauthorised-restoration risk and the permanent-loss risk have no common
    currency until a valuation supplies one. Writing ``C_u`` for the cost of an
    unauthorised restoration and ``C_h`` for that of a written-off vessel and
    minimising ``P_u * C_u + P(loss|c) * C_h``, the two configurations are
    indifferent at

        C_u / C_h  =  (loss_to - loss_from) / (p_u_from - p_u_to).

    ``C_u`` is deliberately not ``C_stack``. An unauthorised restoration hands
    back propulsion and autonomous decision-making to a vessel the adversary
    holds, so it costs at least the stack and at most whatever a hostile party
    can do with a laden ship under its own power. The paper does not price that;
    the omission runs against the recommendation rather than for it, since a
    larger ``C_u`` makes the ratio easier to exceed and argues for a higher ``t``.

    Which side of that ratio favours ``to`` depends on the direction of the
    trade, and the caller must read it off the signs:

    * ``to`` lowers ``P_u`` and raises loss -- ``to`` wins *above* the ratio
      (raising the threshold buys security at the price of recoverability);
    * ``to`` raises ``P_u`` and lowers loss -- ``to`` wins *below* it
      (widening the holder set buys recoverability at a small security price).

    ``0.0`` is returned when ``to`` is better on both axes and ``inf`` when it
    is worse on both, so that no valuation is needed in either case.
    """
    d_u = p_u_from - p_u_to          # unauthorised-restoration risk bought
    d_loss = loss_to - loss_from     # permanent-loss risk paid
    if d_u > 0.0 and d_loss <= 0.0:
        return 0.0                   # dominates
    if d_u < 0.0 and d_loss > 0.0:
        return float("inf")          # dominated
    if d_u == 0.0:
        return 0.0 if d_loss <= 0.0 else float("inf")
    return d_loss / d_u


def gather_time_given_available(k: int, t: int, rho: float) -> float:
    """Mean time to collect ``t`` of ``k`` responses, each at rate ``rho``."""
    if k < t:
        raise ValueError(f"cannot gather t={t} responses from k={k} holders")
    if rho <= 0.0:
        raise ValueError("response rate rho must be positive")
    return sum(1.0 / ((k - j) * rho) for j in range(t))


def expected_gather_time(holders: Sequence[Holder], t: int, rho: float) -> float:
    """E[time to quorum | a quorum exists], averaged over how many holders answer.

    Conditioning matters: the configurations that most often fail to reach a
    quorum are also the ones that would have been slowest, so the unconditional
    mean is not the right input to the chain -- the no-quorum mass is handled
    separately, by the ``Z_u`` state.
    """
    pmf = poisson_binomial_pmf([h.availability for h in holders])
    p_quorum = sum(pmf[t:])
    if p_quorum == 0.0:
        return float("inf")
    weighted = sum(pmf[k] * gather_time_given_available(k, t, rho)
                   for k in range(t, len(pmf)))
    return weighted / p_quorum


def quorum_rate(holders: Sequence[Holder], t: int, rho: float) -> float:
    """The gather stage as a single exponential rate, for the CTMC.

    The true time-to-quorum is a mixture of sums of exponentials, not an
    exponential. Collapsing it to one stage with the matching mean is the
    approximation the paper's five-transient-state chain makes; it preserves the
    mean gather time and therefore E[T_c], while understating the variance.
    """
    mean = expected_gather_time(holders, t, rho)
    return 0.0 if mean == float("inf") else 1.0 / mean
