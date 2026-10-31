"""Reproduction package for *Safety-Gated Reversible Denial: Recoverable
Anti-Tampering for Crewless Merchant Ships*.

Modules
-------
params      Table `tab:params`, plus the published values the paper prints.
threshold   Poisson-binomial DP over heterogeneous holders; quorum timing.
ctmc        The absorbing continuous-time Markov chain of Section `sec:method`.
modelcheck  Exhaustive check of the ten properties over B0 and H1.
asam        Possession duration from the NGA incident database.
report      Recompute everything and diff it against the printed values.
figures     Figs. `fig:regimes` and `fig:frontier`.
refimpl     Tier-0 software reference implementation of the recovery exchange
            and the staged controller (Section `sec:impl`, experiments S1-S6).

Run ``python run_all.py`` for the whole reproduction and
``python run_refimpl.py`` for the reference-implementation experiments.
"""

__all__ = ["params", "threshold", "ctmc", "modelcheck", "asam", "report",
           "figures", "refimpl"]
__version__ = "1.0.0"
