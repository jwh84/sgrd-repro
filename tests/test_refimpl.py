"""Tests for the tier-0 reference implementation.

Separated the same way as ``test_sgrd.py``:

* **constants** -- the embedded group is what it claims to be. A wrong constant
  here would not raise; it would silently move the whole implementation into a
  group with small subgroups, and every functional test would still pass.
* **primitives** -- Shamir, the sigma protocol, the KEM. These would catch an
  implementation bug even if the design were wrong.
* **the design's claims** -- the checks of S1-S5, run as tests so that a
  regression in the controller fails the build rather than quietly changing a
  number in Section `sec:impl`.

Run with ``python tests/test_refimpl.py``. No test framework is required, which
matches ``test_sgrd.py`` and keeps the artefact runnable with nothing installed
beyond ``cryptography``. The full run takes a couple of minutes: the group is
2048-bit and the arithmetic is CPython's.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sgrd.refimpl import experiments as ex  # noqa: E402
from sgrd.refimpl import group as gp  # noqa: E402
from sgrd.refimpl import protocol as pr  # noqa: E402
from sgrd.refimpl import tpre  # noqa: E402

_SCENARIO: pr.Scenario | None = None


def scenario() -> pr.Scenario:
    """Provisioning costs `n + 1` exponentiations, so it is done once and each
    test takes its own copy."""
    global _SCENARIO
    if _SCENARIO is None:
        _SCENARIO = pr.build_scenario()
    return _SCENARIO


@contextmanager
def raises(exc_type, contains: str = ""):
    try:
        yield
    except exc_type as exc:
        assert contains in str(exc), f"expected {contains!r} in {exc}"
    else:
        raise AssertionError(f"expected {exc_type.__name__}, nothing raised")


# --- constants ---------------------------------------------------------------

def test_group_constant_is_what_it_claims():
    """The one test that cannot be replaced by a functional test.

    A modulus that is prime but not *safe* would give the quadratic residues a
    composite order and admit small-subgroup confinement, and every other test
    in this file would still pass.
    """
    for name, ok in gp.self_check().items():
        assert ok, name


def test_membership_rejects_the_obvious_small_subgroup():
    assert gp.is_member(gp.G)
    assert not gp.is_member(gp.P - 1)  # order 2
    assert not gp.is_member(0)
    assert not gp.is_member(gp.P)


def test_encoding_is_fixed_width():
    """Variable-width encoding would make the transcript hash ambiguous."""
    assert len(gp.i2osp(1)) == len(gp.i2osp(gp.P - 1)) == gp.ELEMENT_BYTES


def test_hash_is_length_prefixed():
    """`H(a, bc)` must differ from `H(ab, c)`; otherwise the field boundaries in
    the transcript of `eq:token` are not boundaries."""
    assert (gp.hash_to_scalar(b"d", b"a", b"bc")
            != gp.hash_to_scalar(b"d", b"ab", b"c"))
    assert gp.hash_to_scalar(b"d1", b"x") != gp.hash_to_scalar(b"d2", b"x")


# --- primitives --------------------------------------------------------------

def test_shamir_any_t_reconstructs_and_no_fewer():
    secret = gp.rand_scalar()
    shares = gp.share(secret, 3, 5)
    for sub in combinations(shares, 3):
        assert gp.reconstruct(sub) == secret
    for sub in combinations(shares, 2):
        assert gp.reconstruct(sub) != secret


def test_schnorr_signature_round_trip():
    sk = gp.rand_scalar()
    pk = gp.exp(gp.G, sk)
    sig = pr.sign(sk, b"message")
    assert pr.verify(pk, b"message", sig)
    assert not pr.verify(pk, b"other", sig)
    assert not pr.verify(gp.exp(gp.G, gp.rand_scalar()), b"message", sig)


def test_reshare_proof_rejects_every_single_field_tamper():
    """The sigma protocol has seven public values; altering any one must fail.

    Written per-field rather than as one "tamper with the share" test because a
    proof that checks six of the seven verifies exactly the same on the happy
    path.
    """
    auth, shares = tpre.provision(2, 3)
    cap = tpre.seal(auth, os.urandom(32), vessel_id=b"V", epoch=1)
    pk_eph = gp.exp(gp.G, gp.rand_scalar())
    rs = tpre.reshare(shares[0], cap, pk_eph)
    tpre.verify_reshare(rs, auth, cap, pk_eph)  # control
    for f in ("d", "e", "a", "b", "c"):
        bad = replace(rs, **{f: gp.mul(getattr(rs, f), gp.G)})
        with raises(tpre.ShareRejected):
            tpre.verify_reshare(bad, auth, cap, pk_eph)
    for f in ("z_s", "z_k"):
        bad = replace(rs, **{f: (getattr(rs, f) + 1) % gp.Q})
        with raises(tpre.ShareRejected):
            tpre.verify_reshare(bad, auth, cap, pk_eph)


def test_a_share_is_bound_to_its_capsule_and_ephemeral_key():
    """A share issued for one exchange must not verify in another, or the
    transcript binding of Section `sec:design` is undone at the share level
    rather than at the token level."""
    auth, shares = tpre.provision(2, 3)
    cap1 = tpre.seal(auth, os.urandom(32), vessel_id=b"V", epoch=1)
    cap2 = tpre.seal(auth, os.urandom(32), vessel_id=b"V", epoch=2)
    pk1 = gp.exp(gp.G, gp.rand_scalar())
    pk2 = gp.exp(gp.G, gp.rand_scalar())
    rs = tpre.reshare(shares[0], cap1, pk1)
    with raises(tpre.ShareRejected):
        tpre.verify_reshare(rs, auth, cap2, pk1)
    with raises(tpre.ShareRejected):
        tpre.verify_reshare(rs, auth, cap1, pk2)


def test_the_platform_cannot_open_its_own_capsule():
    """The design's central asymmetry: the platform seals and cannot unseal.

    Were this ever false the argument of Section `sec:design` would collapse,
    since an adversary holding the platform would hold everything.
    """
    auth, _shares = tpre.provision(2, 3)
    kek = os.urandom(32)
    cap = tpre.seal(auth, kek, vessel_id=b"V", epoch=1)
    assert kek not in gp.i2osp(cap.c1) + cap.ct + cap.nonce
    # The only value the platform holds is pk_r, and no function in the module
    # takes pk_r and a capsule to a plaintext.
    assert not hasattr(tpre, "open_with_public_key")


# --- organisational constraints ---------------------------------------------

def test_custody_issuer_may_not_hold_a_share():
    """Section `sec:design` states this and no cryptography enforces it, so the
    only place it can live is a registration-time check."""
    q = pr.build_quorum(2, ["a", "b", "c"], "port-state")
    with raises(ValueError, "disjoint"):
        pr.check_disjoint(replace(q, issuer=replace(q.issuer, name="b")))


def test_holders_must_be_organisationally_distinct():
    with raises(ValueError, "organisationally distinct"):
        pr.build_quorum(2, ["a", "a", "c"], "port-state")


# --- the exchange ------------------------------------------------------------

def test_happy_path_restores_the_working_keys():
    sc = ex._fresh(scenario())
    assert pr.happy_path(sc) == sc.working_keys
    assert sc.platform.phase == pr.NORMAL


def test_denial_leaves_nothing_usable_on_the_platform():
    """Section `sec:design`: what an adversary holding the platform has is
    ciphertext, a wrapped key whose wrapping key no longer exists, and a capsule
    it cannot open."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    sc = ex._fresh(scenario())
    p = sc.platform
    p.deny()
    assert p.kek is None and p.working_keys is None
    assert p.wrapped_keys is not None and p.capsule is not None
    nonce, ct = p.wrapped_keys
    with raises(InvalidTag):
        AESGCM(os.urandom(32)).decrypt(nonce, ct, p.vessel_id)


def test_repeated_episodes_advance_the_counter():
    sc = ex._fresh(scenario())
    for expected in (1, 2, 3):
        pr.happy_path(sc)
        assert sc.platform.counter == expected
        assert sc.platform.epoch == expected + 1


def test_safe_predicate_is_fail_closed():
    """Table `tab:disagree`: every row that is not "both say safe" is false, and
    absence of evidence is one of those rows."""
    for kernel, roc, want in [(True, True, True), (True, False, False),
                              (False, True, False), (False, False, False),
                              (None, True, False), (True, None, False),
                              (None, None, False)]:
        assert pr.SafeEvidence("open-water", kernel, roc).holds() is want
    for situation in ("narrow-channel", "heavy-weather"):
        assert not pr.SafeEvidence(situation, True, True).holds()


# --- the experiments, as regression tests -----------------------------------

def _run_experiment(key: str) -> None:
    """A failure means either the controller regressed or a claim in Section
    `sec:impl` no longer holds; both should break the build."""
    failures = [c for c in ex.EXPERIMENTS[key][1](scenario()) if not c.passed]
    assert not failures, "\n".join(
        f"{c.name}: expected {c.expected}, observed {c.observed}"
        for c in failures)


def test_s1_threshold_assembly():
    _run_experiment("S1")


def test_s2_combiner_ignorance():
    _run_experiment("S2")


def test_s3_transcript_binding():
    _run_experiment("S3")


def test_s4_reuse_and_replay():
    _run_experiment("S4")


def test_s5_rejection_suite():
    _run_experiment("S5")


def test_s3_covers_every_transcript_field():
    """S3 is a claim about completeness, so the count is part of the claim."""
    names = [c.name for c in ex.s3_transcript_binding(scenario())]
    for f in pr.TRANSCRIPT_FIELDS:
        assert any(f"altering {f} " in n for n in names), f
    assert len(pr.TRANSCRIPT_FIELDS) == 9


def test_the_ablation_actually_ablates():
    """S2 is only evidence if the unblinded variant really does leak.

    A test asserting "the combiner cannot open it" passes trivially against an
    implementation where nothing opens anything. This asserts the other
    direction explicitly.
    """
    checks = ex.s2_combiner_ignorance(scenario())
    opened = [c for c in checks
              if "no blinding" in c.name and "cannot open" in c.name]
    assert opened and all(c.observed == "OPENS K_e" for c in opened)


# --- crash consistency of the capsule commit ---------------------------------

def test_slot_record_round_trip():
    from sgrd.refimpl import store as sto
    raw = sto.encode_slot(7, b"wrapped", b"capsule")
    assert sto.decode_slot(raw) == (7, b"wrapped", b"capsule")


def test_a_torn_record_is_rejected_at_every_prefix():
    """Every truncation must fail, not just the obvious ones. A parser that
    reads the length fields before checking the digest can be walked off the
    end of a short buffer, and that is a crash rather than a rejection."""
    from sgrd.refimpl import store as sto
    raw = sto.encode_slot(7, b"w" * 40, b"c" * 90)
    for cut in range(len(raw)):
        assert sto.decode_slot(raw[:cut]) is None, cut
    assert sto.decode_slot(raw) is not None


def test_counter_records_alternate():
    """Regression: the first version chose the counter record by epoch parity,
    which overwrites the only valid record every second commit. A write torn
    there destroyed the store outright, and experiment A2 found it. The rule is
    now "write the older record", and this pins it.
    """
    from sgrd.refimpl import store as sto
    dev = sto.SimDevice()
    store = sto.CapsuleStore(dev)
    store.provision(b"w1", b"c1")
    seen = []
    for e in range(2, 8):
        before = dict(dev.durable)
        store.commit(f"w{e}".encode(), f"c{e}".encode())
        written = [n for n in sto.COUNTERS if dev.durable.get(n) != before.get(n)]
        seen.append(written[0])
        # At every instant, at least one counter record is valid and names a
        # slot that exists.
        assert store.recover().counter == e
    assert seen == ["ctr1", "ctr0"] * 3, seen


def test_clean_commits_leave_one_coherent_pair():
    from sgrd.refimpl import store as sto
    dev = sto.SimDevice()
    store = sto.CapsuleStore(dev)
    store.provision(b"w1", b"c1")
    for e in range(2, 6):
        store.commit(f"w{e}".encode(), f"c{e}".encode())
    state = store.recover()
    assert (state.counter, state.epoch) == (5, 5)
    assert state.wrapped == b"w5" and state.capsule == b"c5"
    assert not state.rolled_back
    assert store.recover() == state          # idempotent


def test_recovery_refuses_an_empty_store():
    from sgrd.refimpl import store as sto
    with raises(sto.Uninitialised):
        sto.CapsuleStore(sto.SimDevice()).recover()


def _run_crash(key: str) -> None:
    from sgrd.refimpl import crash_experiments as cx
    ep = cx.build_epochs()
    failures = [c for c in cx.CRASH_EXPERIMENTS[key][1](ep) if not c.passed]
    assert not failures, "\n".join(
        f"{c.name}: expected {c.expected}, observed {c.observed} [{c.detail}]"
        for c in failures)


def test_a1_power_cut():
    _run_crash("A1")


def test_a2_torn_writes():
    _run_crash("A2")


def test_a3_sync_ablation():
    _run_crash("A3")


def test_a4_real_process_kill():
    _run_crash("A4")


def test_a3_finds_the_syncs_that_matter():
    """A3 is only evidence if it distinguishes. An ablation that called every
    sync load-bearing, or none, would be measuring nothing."""
    from sgrd.refimpl import crash_experiments as cx
    checks = cx.a3_sync_ablation(cx.build_epochs())
    verdicts = [c.observed for c in checks if c.name.startswith("A3 sync ")]
    assert "load-bearing" in verdicts
    assert "no effect in this sweep" in verdicts


# --- tier 1: the TPM ---------------------------------------------------------

class Skipped(Exception):
    """No suitable hardware. Reported separately from a pass and from a fail."""


def _require_tpm():
    from sgrd.refimpl import tpm_experiments as tx
    status = tx.hardware_status()
    if not status["tpm"].get("available"):
        raise Skipped(f"no usable TPM: {status['tpm'].get('reason')}")
    if not status["platform_crypto_provider"].get("available"):
        raise Skipped("no TPM-backed CNG provider: "
                      f"{status['platform_crypto_provider'].get('reason')}")
    return tx


def test_pcr_selection_bitmap_is_little_endian_per_byte():
    """The selection bitmap is the easiest thing here to get backwards, and
    getting it backwards selects the wrong PCRs rather than erroring."""
    from sgrd.refimpl import tpm as tp
    assert tp.pcr_selection([0])[-3:] == bytes([0x01, 0x00, 0x00])
    assert tp.pcr_selection([7])[-3:] == bytes([0x80, 0x00, 0x00])
    assert tp.pcr_selection([8])[-3:] == bytes([0x00, 0x01, 0x00])
    assert tp.pcr_selection([23])[-3:] == bytes([0x00, 0x00, 0x80])
    assert tp.pcr_selection(range(8))[-3:] == bytes([0xFF, 0x00, 0x00])
    with raises(ValueError, "out of range"):
        tp.pcr_selection([24])


def test_policy_digest_depends_on_every_selected_pcr():
    """If it did not, sealing to a state would not bind to all of it."""
    from sgrd.refimpl import tpm as tp
    base = {i: bytes([i + 1]) * 32 for i in range(8)}
    ref = tp.policy_pcr_digest(base)
    assert len(ref) == 32
    assert tp.policy_pcr_digest(dict(base)) == ref  # deterministic
    for i in base:
        altered = dict(base)
        altered[i] = bytes(32)          # distinct from every base value
        assert altered[i] != base[i]
        assert tp.policy_pcr_digest(altered) != ref, i


def test_policy_fail_is_recognised_under_its_qualifiers():
    """TPM format-one codes carry a handle, parameter or session number in the
    upper bits; comparing the code whole would miss the qualified variants."""
    from sgrd.refimpl import tpm as tp
    assert tp.is_policy_fail(0x09D)
    assert tp.is_policy_fail(0x89D)   # with a session qualifier
    assert not tp.is_policy_fail(0x000)
    assert not tp.is_policy_fail(0x09C)


def test_hardware_status_never_raises():
    """Callers use this to decide whether to run tier 1 at all, so it has to be
    safe on a machine with no TPM, no drivers, or no Windows."""
    from sgrd.refimpl import tpm_experiments as tx
    status = tx.hardware_status()
    assert set(status) == {"tpm", "platform_crypto_provider"}
    for v in status.values():
        assert "available" in v


def test_t1_device_bound_secret():
    tx = _require_tpm()
    failures = [c for c in tx.t1_device_bound_secret() if not c.passed]
    assert not failures, "\n".join(f"{c.name}: {c.observed}" for c in failures)


def test_t2_measured_state():
    tx = _require_tpm()
    failures = [c for c in tx.t2_measured_state() if not c.passed]
    assert not failures, "\n".join(f"{c.name}: {c.observed}" for c in failures)


def test_t3_sealing_refuses_on_a_changed_measurement():
    """The load-bearing tier-1 result, as a test.

    Asserted in both directions: the control must open, or "refused" would be
    the answer to every question and the experiment would prove nothing.
    """
    tx = _require_tpm()
    checks = tx.t3_sealing_to_measured_state()
    failures = [c for c in checks if not c.passed]
    assert not failures, "\n".join(f"{c.name}: {c.observed}" for c in failures)
    opened = [c for c in checks if c.observed == "opened"]
    refused = [c for c in checks if c.observed == "refused"]
    assert len(opened) >= 2, "the control did not open"
    assert len(refused) >= 4, "nothing was refused"


def test_t4_monotonic_values():
    tx = _require_tpm()
    failures = [c for c in tx.t4_monotonic_values() if not c.passed]
    assert not failures, "\n".join(f"{c.name}: {c.observed}" for c in failures)


def test_t5_exchange_with_hardware_inputs():
    tx = _require_tpm()
    failures = [c for c in tx.t5_hardware_backed_exchange(scenario())
                if not c.passed]
    assert not failures, "\n".join(f"{c.name}: {c.observed}" for c in failures)


def test_tier1_leaves_no_persistent_tpm_state():
    """The experiments must not change the machine.

    PCR 23 is read before and after: it is the only PCR anything here would
    write, and the write is blocked in any case.
    """
    tx = _require_tpm()
    from sgrd.refimpl import tpm as tp
    with tp.Tpm() as t:
        before = t.pcr_read([tp.APPLICATION_PCR])[tp.APPLICATION_PCR]
    tx.t3_sealing_to_measured_state()
    with tp.Tpm() as t:
        after = t.pcr_read([tp.APPLICATION_PCR])[tp.APPLICATION_PCR]
    assert before == after == bytes(32)


def test_latency_is_far_below_the_ctmc_timescale():
    """The quantitative point of S6, as an assertion rather than a table.

    Section `sec:results` puts the mean restoration time at about 23 hours,
    dominated by quorum gathering. Were the cryptographic path ever to approach
    that, the CTMC parameterisation would need revisiting -- so the test is
    written as the condition under which Section `sec:results` still stands.
    """
    timings = ex.s6_latency_budget(scenario(), n=3)
    episode = next(t for t in timings
                   if t.operation.startswith("full episode"))
    assert episode.mean_ms < 60_000, episode  # one minute, against 23 hours


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in fns:
        try:
            fn()
        except Skipped as exc:
            skipped += 1
            print(f"skip {name}\n     {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {name}\n     {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    print(f"\n{len(fns) - failed - skipped}/{len(fns)} tests passed"
          + (f", {skipped} skipped for want of hardware" if skipped else ""))
    sys.exit(1 if failed else 0)
