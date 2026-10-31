"""A software reference implementation of the SGRD recovery path.

Section `sec:impl` divides the validation into tiers by what equipment each
needs. This package is tier 0: everything that can be settled with a general
purpose computer and no security hardware. It is deliberately *not* the tier
that would establish the design's hardware claims -- zeroisation completeness,
PUF error rates, and power-loss atomicity of the capsule commit all live above
it, and Section `sec:disc` says so.

What tier 0 can settle is the part that is arithmetic and control flow:

===  ==================================================================
S1   The threshold re-encryption path assembles, and does not assemble
     below `t`.
S2   The combiner learns nothing -- shown by ablation against the
     unblinded construction, which lets the combiner recover `K_e`.
S3   All nine transcript bindings of `eq:token` hold, one field at a time.
S4   Consumed, replayed, pre-minted, and cross-episode tokens are refused.
S5   The nine rejection tests of Table `tab:impl-reject`.
S6   Where the time goes, so that Section `sec:results`'s critical path
     can be checked against the operations it assumes are negligible.
===  ==================================================================

The numbering is `S` and not `E` on purpose: Table `tab:impl-scope` already uses
`E1..E6` for the six components of the reference configuration `C_0`, and these
are experiments over those components, not a renumbering of them.

Run ``python run_refimpl.py`` from the repository root.
"""

__all__ = ["group", "tpre", "protocol", "experiments"]
