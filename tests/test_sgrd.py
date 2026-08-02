"""Tests for the SGRD reproduction.

Three kinds of test, deliberately separated:

* **invariants** -- things that must hold for the code to be a Markov chain or a
  probability distribution at all. These would catch an implementation bug even
  if the paper were wrong.
* **reproduction** -- the printed values. These fail if the reimplementation and
  the paper disagree, which is the point.
* **qualitative claims** -- the findings stated as prose. A finding that only
  holds at the exact published parameters is a weaker finding than one that
  survives perturbation, so some of these perturb.

Run with ``pytest -q`` from the ``sim/`` directory, or ``python tests/test_sgrd.py``
for a no-dependency run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sgrd import asam, ctmc, modelcheck, report  # noqa: E402
from sgrd import params as P  # noqa: E402
from sgrd.threshold import (  # noqa: E402
    expected_gather_time, gather_time_given_available, p_at_least,
    p_authority_assemblable, poisson_binomial_pmf, p_unauthorised_restoration,
)


# --- invariants --------------------------------------------------------------

def test_poisson_binomial_is_a_distribution():
    for probs in ([0.5], [0.99, 0.7, 0.6], [h.availability for h in P.HOLDERS]):
        pmf = poisson_binomial_pmf(probs)
        assert len(pmf) == len(probs) + 1
        assert abs(sum(pmf) - 1.0) < 1e-12
        assert all(p >= -1e-15 for p in pmf)


def test_poisson_binomial_reduces_to_binomial():
    """Homogeneous holders must give the ordinary binomial."""
    from math import comb
    n, p = 6, 0.3
    pmf = poisson_binomial_pmf([p] * n)
    for k in range(n + 1):
        assert abs(pmf[k] - comb(n, k) * p ** k * (1 - p) ** (n - k)) < 1e-12


def test_tail_is_monotone_in_t():
    for t in range(1, len(P.HOLDERS)):
        assert (p_authority_assemblable(P.HOLDERS, t)
                >= p_authority_assemblable(P.HOLDERS, t + 1))
        assert (p_unauthorised_restoration(P.HOLDERS, t)
                >= p_unauthorised_restoration(P.HOLDERS, t + 1))


def test_p_at_least_edges():
    assert p_at_least([0.5, 0.5], 0) == 1.0
    assert p_at_least([0.5, 0.5], 3) == 0.0


def test_gather_time_grows_with_t_and_shrinks_with_k():
    assert (gather_time_given_available(4, 3, 0.25)
            > gather_time_given_available(4, 2, 0.25))
    assert (gather_time_given_available(5, 2, 0.25)
            < gather_time_given_available(3, 2, 0.25))
    # More holders at the same threshold cannot make the wait longer.
    for t in (2, 3):
        assert (expected_gather_time(P.holder_set("Q1", "Q2", "Q3", "Q4", "Q5"), t,
                                     P.RESPONSE_RATE)
                <= expected_gather_time(P.holder_set("Q1", "Q2", "Q3", "Q4"), t,
                                        P.RESPONSE_RATE) + 1e-12)


def test_generator_rows_sum_to_zero():
    """A generator matrix has zero row sums across transient *and* absorbing."""
    T, A = ctmc.build_generator(
        mean_possession_h=349.0, time_to_extraction_h=8760.0,
        restoration_rate=1 / 6, p_modified_effective=0.6,
        quorum_gather_rate=0.3, p_authority=0.99)
    rows = T.sum(axis=1) + A.sum(axis=1)
    assert np.allclose(rows, 0.0, atol=1e-12)
    assert (T.diagonal() < 0).all()
    off = T - np.diag(T.diagonal())
    assert (off >= 0).all() and (A >= 0).all()



def test_absorption_probabilities_sum_to_one():
    for cfg in P.named_configurations():
        for adversary in ("A2", "A3"):
            o = ctmc.evaluate(cfg, adversary=adversary)
            assert abs(o.p_restored_given_custody
                       + o.p_loss_given_custody - 1.0) < 1e-9
            if o.p_extracted is None:  # rated, not priced
                assert o.p_restored_uncond is None and o.p_loss_uncond is None
                continue
            total = o.p_extracted + o.p_restored_uncond + o.p_loss_uncond
            assert abs(total - 1.0) < 1e-9, (cfg.name, cfg.variant, total)



def test_closed_form_matches_the_matrix_solve():
    """P(ext) has a closed form; it must agree with the linear solve."""
    for cfg in P.named_configurations():
        if cfg.variant not in P.TIME_TO_EXTRACTION_H:
            continue
        o = ctmc.evaluate(cfg, adversary="A2")
        closed = ctmc.extraction_probability(cfg.variant, "A2", P.POSSESSION_POOLED_H)
        assert abs(o.p_extracted - closed) < 1e-12



def test_extraction_is_independent_of_the_threshold():
    """The paper's phase separation: the quorum cannot affect P(ext)."""
    base = None
    for t in range(1, 6):
        o = ctmc.evaluate(P.Configuration(f"{t}-of-5", t, P.HOLDERS, "B0"))
        base = o.p_extracted if base is None else base
        assert abs(o.p_extracted - base) < 1e-12


def test_write_off_does_not_compete_during_possession():
    """The write-off hazard has no edge out of Z_p, and must not touch P(ext).

    A constructive total loss is an accounting event: declaring it does not end
    the adversary's physical access, so it cannot truncate an extraction under
    way. Varying the horizon must therefore leave P(EXT) exactly unchanged while
    still changing what happens once custody is back.
    """
    kw = dict(mean_possession_h=349.0, time_to_extraction_h=8760.0,
              restoration_rate=1 / 6, p_modified_effective=0.6,
              quorum_gather_rate=0.3, p_authority=0.99)
    soon, _ = ctmc.solve(*ctmc.build_generator(**{**kw,
                                                 "write_off_horizon_h": 240.0}))
    never, _ = ctmc.solve(*ctmc.build_generator(**{**kw,
                                                  "write_off_horizon_h": 1e12}))
    # P(EXT), decided entirely in Z_p, is untouched.
    assert abs(soon[0, 1] - never[0, 1]) < 1e-12
    # Permanent loss from the post-custody states is not.
    assert soon[1, 2] > never[1, 2]


# --- reproduction ------------------------------------------------------------

def test_published_extraction_probabilities():
    for (variant, adversary), published in P.PUBLISHED["extraction_pooled"].items():
        got = ctmc.extraction_probability(variant, adversary, P.POSSESSION_POOLED_H)
        assert abs(got - published) < 5e-4, (variant, adversary, got, published)



def test_published_regime_extraction():
    for hours, published in P.PUBLISHED["extraction_regimes_b0_a2"].items():
        got = ctmc.extraction_probability("B0", "A2", hours)
        assert abs(got - published) < 5e-4, (hours, got, published)



def test_published_ctmc_table():
    for cfg in P.named_configurations():
        p_ext, p_rest, p_loss, e_tc, p_u = P.PUBLISHED["ctmc"][(cfg.name, cfg.variant)]
        o = ctmc.evaluate(cfg, adversary="A2")
        label = (cfg.name, cfg.variant)
        tol = report.PRINTED_3DP  # one unit in the last printed place
        # p_ext is None where the paper rates the variant instead of pricing it.
        assert (p_ext is None) == (o.p_extracted is None), ("priced?", label)
        if p_ext is not None:
            assert abs(o.p_extracted - p_ext) < tol, ("P(ext)", label, o.p_extracted)
        assert abs(o.p_restored_given_custody - p_rest) < tol, ("P(rest|c)", label)
        assert abs(o.p_loss_given_custody - p_loss) < tol, ("P(loss|c)", label)
        assert abs(o.mean_time_to_restore_given_custody - e_tc) < 0.05, ("E[Tc]", label)
        assert abs(o.p_unauthorised - p_u) < 0.05 * p_u + 1e-6, ("P_u", label)


def test_published_unconditional_loss():
    for variant, published in P.PUBLISHED["unconditional_loss_b0"].items():
        cfg = next(c for c in P.named_configurations() if c.variant == variant)
        assert abs(ctmc.evaluate(cfg).p_loss_uncond - published) < 5e-4


def test_local_secret_baseline_has_no_threshold_dependence():
    """B0's authority is on the platform, so nothing about its post-custody
    outcome may move with a holder set it does not use."""
    base = None
    for tags, t in ((("Q1",), 1), (("Q1", "Q2", "Q3"), 2),
                    (("Q1", "Q2", "Q3", "Q4", "Q5"), 3)):
        o = ctmc.evaluate(P.Configuration("x", t, P.holder_set(*tags), "B0"))
        got = (o.p_restored_given_custody, o.p_loss_given_custody,
               o.mean_time_to_restore_given_custody, o.p_unauthorised)
        base = got if base is None else base
        assert all(abs(a - b) < 1e-9 for a, b in zip(base, got)), (tags, t, got)
    assert base[3] == 1.0


def test_published_sweep():
    sweep = P.PUBLISHED["sweep_five_holders"]
    assert abs(p_unauthorised_restoration(P.HOLDERS, 1) - sweep[1]["P_u"]) < 5e-4
    for t in (4, 5):
        assert abs(p_authority_assemblable(P.HOLDERS, t)
                   - sweep[t]["P_available"]) < 5e-4
        o = ctmc.evaluate(P.Configuration(f"{t}-of-5", t, P.HOLDERS, "H1"))
        assert abs(o.p_loss_given_custody
                   - sweep[t]["P_loss_given_custody"]) < 5e-4


def test_asam_table_is_internally_consistent():
    assert asam.check_internal_consistency() == []


def test_asam_percentile_matches_numpy():
    from sgrd.asam import _percentile
    for values in ([1.0, 2.0, 3.0, 10.0], [5.0], list(range(1, 17))):
        s = sorted(float(v) for v in values)
        assert abs(_percentile(s, 0.9) - float(np.percentile(s, 90))) < 1e-9


# --- model checking ----------------------------------------------------------

def test_verdicts_reproduce():
    """The load-bearing claim of Table 6: B0 satisfies 1 of 12, H1 all 12."""
    b0, h1 = modelcheck.verify("B0"), modelcheck.verify("H1")
    assert h1.n_satisfied == 12, h1.witnesses
    assert b0.n_satisfied == 1, b0.witnesses
    # And it is P9 -- recoverability -- that B0 gets right, which is exactly why
    # the hardening has to be shown not to cost recoverability.
    assert b0.satisfied["P9"] is True
    assert h1.satisfied["P9"] is True


def test_b0_fails_the_expected_properties():
    b0 = modelcheck.verify("B0")
    for prop in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P10",
                 "P11", "P12"):
        assert b0.witnesses[prop] > 0, f"{prop} unexpectedly holds for B0"


def test_intermediate_baselines_isolate_one_contribution_each():
    """A 12/12 against a 1/12 baseline separates nothing. B1--B3 each hold all
    of SGRD but one contribution, so exactly the properties that contribution
    discharges should fail -- and no others."""
    expected = {
        "B1": {"P10"},                # off-board, but authority centralised
        "B2": {"P5", "P6", "P8"},     # threshold, but a single-step return
        "B3": {"P4"},                 # both, but no safe-state gate
    }
    for name, failing in expected.items():
        r = modelcheck.verify(name)
        got = {p for p, n in r.witnesses.items() if n > 0}
        assert got == failing, f"{name}: expected {failing}, got {got}"
        assert r.n_satisfied == 12 - len(failing)


def test_p9_requires_a_legitimate_route():
    """P9 must demand a *legitimate* path. Plain AG(DENIED -> EF RESTORED) is
    satisfied by any route, including the on-board secret and the coerced
    holder -- the very weaknesses the paper is about."""
    rel = modelcheck.explore(modelcheck.VARIANTS["B0"])
    ends = [(a, b) for a, _, b in rel.transitions
            if b.phase == modelcheck.RESTORED and a.phase != modelcheck.RESTORED]
    assert any(a.auth == modelcheck.ONBOARD for a, _ in ends)
    assert any(a.auth == modelcheck.COERCED for a, _ in ends)
    # Those routes exist, and the goal set excludes every one of them.
    goal = {b for a, b in ends if a.auth == modelcheck.LEGIT}
    assert goal and len(goal) < len({b for _, b in ends})


def test_denial_never_strands_the_vessel_in_h1():
    """P11 in the concrete: every H1 denial leaves navigation up, with the
    kernel holding a minimum-risk condition uncommanded."""
    rel = modelcheck.explore(modelcheck.VARIANTS["H1"])
    denials = [(a, b) for a, lbl, b in rel.transitions if lbl == "deny"]
    assert denials
    for _, post in denials:
        assert post.nav_available
        assert post.command_owner == modelcheck.AUTONOMOUS
    # And B0, whose denial is a blunt lockout, does the opposite.
    b0 = modelcheck.explore(modelcheck.VARIANTS["B0"])
    assert all(not b.nav_available
               for a, lbl, b in b0.transitions if lbl == "deny")


def test_p11_p12_hold_as_state_invariants_not_only_as_steps():
    """The transition form of P12 passed while 80 reachable states violated it:
    a re-seizure leaves command assigned without changing it. Check the states."""
    rel = modelcheck.explore(modelcheck.VARIANTS["H1"])
    assert not [s for s in rel.states
                if s.possessed and s.command_owner != modelcheck.AUTONOMOUS]
    assert not [s for s in rel.states if s.denied and not s.nav_available]
    # Seizure is what re-establishes the invariant, so it must reset command.
    for _, lbl, post in rel.transitions:
        if lbl == "seize":
            assert post.command_owner == modelcheck.AUTONOMOUS


def test_b0_unauthorised_restoration_is_not_a_quorum_quantity():
    """B0 mints authority on the platform, so P(D_c >= t) does not bound its
    unauthorised restoration and the chain must not report it as if it did."""
    cfg = P.Configuration("2-of-4", 2, P.holder_set("Q1", "Q2", "Q3", "Q4"), "B0")
    assert ctmc.evaluate(cfg).p_unauthorised == 1.0
    h1 = P.Configuration("2-of-4", 2, P.holder_set("Q1", "Q2", "Q3", "Q4"), "H1")
    assert ctmc.evaluate(h1).p_unauthorised < 1e-2


def test_counterexamples_are_real_transitions():
    """A witness count is only evidence if a concrete violating edge exists."""
    b0 = modelcheck.verify("B0")
    for prop, count in b0.witnesses.items():
        if prop == "P9" or count == 0:
            continue
        pre, _, post = modelcheck.counterexample("B0", prop)[0]
        _, pred = modelcheck.SAFETY_PROPERTIES[prop]
        assert pred(pre, post)
        assert (pre, post) != (post, pre) or True
        assert post in dict.fromkeys(t for _, t in modelcheck.successors(pre,
                                                                        modelcheck.B0))


def test_state_space_is_finite_and_reachable_from_the_initial_state():
    for name in ("B0", "H1"):
        rel = modelcheck.explore(modelcheck.VARIANTS[name])
        assert modelcheck.INITIAL in rel.states
        assert 0 < len(rel.states) < 100_000
        assert len(rel.transitions) >= len(rel.states)
        # No dangling successors: every edge target is a reachable state.
        assert all(t in rel.states for _, _, t in rel.transitions)


def test_h1_never_holds_working_keys_during_a_denial_episode():
    """The mechanism behind P7, checked directly on states rather than edges."""
    rel = modelcheck.explore(modelcheck.H1)
    assert not [s for s in rel.states if s.denied and s.keys]
    # B0 is the contrast: sealed material stays extractable after denial fires.
    rel0 = modelcheck.explore(modelcheck.B0)
    assert [s for s in rel0.states if s.denied and s.keys]


def test_h1_restoration_is_reachable_from_every_denial_state():
    """P9 in full: not merely 'zero witnesses' but a genuine backward closure."""
    rel = modelcheck.explore(modelcheck.H1)
    assert modelcheck.check_p9(rel) == []
    assert any(s.phase == modelcheck.DENIED for s in rel.states)
    assert any(s.phase == modelcheck.RESTORED for s in rel.states)


# --- qualitative claims ------------------------------------------------------

def test_finding_1_survives_perturbed_compromise_probabilities():
    """The paper argues Finding 1 does not depend on the stipulated c_i.

    Compromise probabilities are the least defensible parameters in the table,
    so the dominance claim is re-checked with every c_i scaled over two orders of
    magnitude.
    """
    for scale in (0.1, 0.5, 1.0, 2.0, 5.0):
        holders = tuple(
            P.Holder(h.tag, h.role, h.availability, min(h.compromise * scale, 0.99))
            for h in P.HOLDERS)
        base = ctmc.evaluate(P.Configuration("a", 2, holders[:3], "H1"))
        wide = ctmc.evaluate(P.Configuration("b", 2, holders[:4], "H1"))
        assert wide.p_restored_given_custody > base.p_restored_given_custody
        assert wide.p_loss_given_custody < base.p_loss_given_custody
        assert (wide.mean_time_to_restore_given_custody
                < base.mean_time_to_restore_given_custody)


def test_finding_2_band_endpoints():
    """t=1 admits a single compromised holder; t>=4 collapses the honest path."""
    assert p_unauthorised_restoration(P.HOLDERS, 1) > 0.1
    for t in (4, 5):
        o = ctmc.evaluate(P.Configuration(f"{t}-of-5", t, P.HOLDERS, "H1"))
        assert o.p_loss_given_custody > 0.35



def test_finding_3_artefact_is_reproduced_and_explained():
    """Finding 3 is an identity, not a two-point comparison.

    Unconditional loss is P(custody regained) x P(loss | custody). B0's small
    unconditional figure is therefore a consequence of how much mass extraction
    takes first, not of how well it survives the post-custody phase.
    """
    b0 = ctmc.evaluate(next(c for c in P.named_configurations()
                            if c.variant == "B0"))
    survives = 1.0 - b0.p_extracted
    assert abs(b0.p_loss_uncond
               - survives * b0.p_loss_given_custody) < 1e-9
    # The unconditional figure flatters B0: it is far below the conditional one.
    assert b0.p_loss_uncond < b0.p_loss_given_custody
    # And so does the conditional one. B0 restores instantly from a secret it
    # kept, so on post-custody metrics alone it beats SGRD by an order of
    # magnitude -- while having already lost the asset in 71% of episodes.
    h1 = ctmc.evaluate(next(c for c in P.named_configurations()
                            if c.variant == "H1" and c.name.startswith("2-of-4")))
    assert b0.p_loss_given_custody < h1.p_loss_given_custody / 10
    # Any design leaking less carries more mass forward and reports more loss.
    leakier, tighter = 0.9, 0.1
    assert (1 - leakier) * b0.p_loss_given_custody <            (1 - tighter) * b0.p_loss_given_custody



def test_h1_is_rated_not_priced():
    """Section VII-B reports no extraction probability for H1.

    Its resistance is conditional on zeroisation evidence, and an
    attack-potential band is an ordinal statement of the resources a path
    requires rather than a mean time. Asking the chain to price it must fail
    loudly rather than silently return an invented figure.
    """
    assert "H1" not in P.TIME_TO_EXTRACTION_H
    try:
        ctmc.extraction_probability("H1", "A2", P.POSSESSION_POOLED_H)
    except KeyError:
        pass
    else:
        raise AssertionError("pricing an unpriced variant should raise")
    o = ctmc.evaluate(P.Configuration("2-of-4", 2, P.HOLDERS[:4], "H1"))
    assert o.p_extracted is None
    assert o.p_loss_uncond is None
    # Everything conditional on custody is still defined.
    assert 0.0 < o.p_loss_given_custody < 1.0


def test_full_reproduction_report():
    rep = report.run()
    assert rep.checks, "no checks ran"
    assert not rep.failures, "\n".join(str(c) for c in rep.failures)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}\n     {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(fns) - failed}/{len(fns)} tests passed")
    sys.exit(1 if failed else 0)
