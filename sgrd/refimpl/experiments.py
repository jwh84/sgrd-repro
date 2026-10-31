"""Tier-0 experiments S1-S6 (Section `sec:impl`).

Every experiment here is a *rejection* test or an ablation, with one exception:
S6, which measures. That bias is deliberate. A functional test says the design
does what it was built to do, which is the claim least in doubt. What Table
`tab:impl-reject` asks is what the controller declines to do, and a refusal is
only evidence if it is refused *for the stated reason* -- so every check below
asserts the reason string, not merely that an exception was raised.

Each experiment returns a list of :class:`Check`, and a check is `passed` when
the observed reason matches the expected one. A check that raised the right
exception for the wrong reason fails, loudly.
"""

from __future__ import annotations

import copy
import os
import statistics
import time
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from typing import Callable, Sequence

from cryptography.exceptions import InvalidTag

from . import group as gp
from . import protocol as pr
from . import tpre


@dataclass
class Check:
    """One assertion, with what was expected and what happened."""

    experiment: str
    name: str
    expected: str
    observed: str
    passed: bool
    note: str = ""
    #: What the controller said beyond the reason code -- for a transcript
    #: mismatch, which fields differed. Recording it is what turns "refused"
    #: into evidence about *which* binding did the refusing.
    detail: str = ""


@dataclass
class Timing:
    operation: str
    n: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float


def _expect_refusal(experiment: str, name: str, expected_reason: str,
                    fn: Callable[[], object], note: str = "") -> Check:
    """Run `fn`, require it to refuse, and require the reason to match.

    `expected_reason` may name a prefix, so that a reason carrying a detail
    (`transcript_mismatch` with the differing fields) still matches.
    """
    detail = ""
    try:
        fn()
    except pr.Refused as exc:
        observed, detail = exc.reason, exc.detail
    except tpre.ShareRejected as exc:
        observed, detail = f"share_rejected/{exc.reason}", f"holder {exc.index}"
    except InvalidTag:
        observed = "InvalidTag"
    except Exception as exc:  # noqa: BLE001 -- an unexpected type is a failure
        observed = f"unexpected:{type(exc).__name__}:{exc}"
    else:
        observed = "accepted"
    return Check(experiment, name, expected_reason, observed,
                 observed == expected_reason, note, detail)


def _expect_success(experiment: str, name: str, fn: Callable[[], bool],
                    note: str = "") -> Check:
    try:
        ok = fn()
        observed = "accepted" if ok else "accepted-but-wrong-value"
    except Exception as exc:  # noqa: BLE001
        observed = f"{type(exc).__name__}:{exc}"
    return Check(experiment, name, "accepted", observed,
                 observed == "accepted", note)


# --- a reusable episode ------------------------------------------------------

@dataclass
class Episode:
    """A denial episode driven up to, but not through, the R0 gate.

    Holding the pieces separately is what makes the perturbation experiments
    possible: S3 alters one transcript field, S4 substitutes a token from
    another episode, and S5 withholds the custody attestation, all against the
    same otherwise-valid exchange.
    """

    scenario: pr.Scenario
    request: pr.RecoveryRequest
    custody: pr.CustodyAttestation
    transcript: pr.Transcript
    token: pr.Token
    reencrypted: tpre.ReEncrypted
    signers: tuple[int, ...]


def open_episode(sc: pr.Scenario, *, situation: str = "open-water",
                 kernel: bool | None = True, roc: bool | None = True,
                 signers: Sequence[int] | None = None,
                 counter_override: int | None = None,
                 blind: bool = True,
                 reshare_signers: Sequence[int] | None = None) -> Episode:
    p = sc.platform
    if not p.denied:
        p.deny()
    idx = tuple(signers) if signers is not None else tuple(
        h.index for h in sc.quorum.holders[:sc.quorum.authority.t])
    req = p.request_recovery(pr.SafeEvidence(situation, kernel, roc))
    cust = sc.quorum.issuer.issue(
        vessel_id=p.vessel_id, counter=p.counter, nonce=p.nonce,
        boarding_ref=b"boarding-2026-014", issued_at=p.clock())
    t = replace(req.transcript, h_cust=cust.digest())
    if counter_override is not None:
        t = replace(t, counter=counter_override)
    token = pr.issue_token(sc.quorum, t, idx)
    re = pr.reencrypt(sc.quorum, req.capsule, p.pk_eph,
                      reshare_signers if reshare_signers is not None else idx,
                      blind=blind)
    return Episode(sc, req, cust, t, token, re, idx)


def _fresh(base: pr.Scenario) -> pr.Scenario:
    """A private copy, so one experiment cannot leave state for the next."""
    return copy.deepcopy(base)


def current_custody(sc: pr.Scenario) -> pr.CustodyAttestation:
    """A valid custody attestation for whatever episode the platform is in now.

    Needed because `A_cust` carries the same `nu` and `c` the token binds to,
    which makes it an *independent* replay barrier: hand a replayed token its
    original attestation and the attestation is refused before the token is
    examined. To test what the token's own binding does, the adversary has to be
    granted the strongest thing it could plausibly have -- a currently valid
    custody attestation -- and refused anyway.
    """
    p = sc.platform
    return sc.quorum.issuer.issue(
        vessel_id=p.vessel_id, counter=p.counter, nonce=p.nonce,
        boarding_ref=b"boarding-2026-014", issued_at=p.clock())


# --- S1: the threshold path assembles, and only at t -------------------------

def s1_threshold_assembly(base: pr.Scenario) -> list[Check]:
    """Requirements 1 and 3 of Section `sec:design`.

    The interesting half is not that `t` shares open the capsule but that every
    `t`-subset does and no `t-1`-subset does. A construction that works for the
    first `t` holders and not for the last is a construction whose availability
    argument in Section `sec:results` does not hold, since that argument is
    about *any* `t` of `n` being reachable.
    """
    out: list[Check] = []
    sc = _fresh(base)
    q, p = sc.quorum, sc.platform
    t, n = q.authority.t, q.authority.n
    all_idx = [h.index for h in q.holders]

    p.deny()
    req = p.request_recovery(pr.SafeEvidence("open-water", True, True))
    cap, pk_eph, sk_eph = req.capsule, p.pk_eph, p.sk_eph
    kek_true = None  # the platform's K_e is gone; recover it and compare below

    for subset in combinations(all_idx, t):
        re = pr.reencrypt(q, cap, pk_eph, subset)
        got = tpre.unwrap(re, sk_eph, cap)
        if kek_true is None:
            kek_true = got
        out.append(Check(
            "S1", f"open with holders {subset}", "accepted",
            "accepted" if got == kek_true else "accepted-but-wrong-value",
            got == kek_true,
            f"all {len(list(combinations(all_idx, t)))} of the {t}-subsets"))

    for subset in combinations(all_idx, t - 1):
        out.append(_expect_refusal(
            "S1", f"refuse with holders {subset}", "InvalidTag",
            lambda s=subset: tpre.unwrap(
                pr.reencrypt(q, cap, pk_eph, s), sk_eph, cap),
            "below threshold: an authentication failure, not a partial answer"))

    # Requirement 3: a malformed share is attributed, not merely fatal.
    shares = [tpre.reshare(q.by_index[i].share, cap, pk_eph) for i in all_idx[:t]]
    corrupted = replace(shares[1], d=gp.mul(shares[1].d, gp.G))
    out.append(_expect_refusal(
        "S1", "malformed share is attributed to its holder",
        f"share_rejected/proof failed on D_i, E_i, V_i",
        lambda: tpre.combine([shares[0], corrupted, shares[2]],
                             q.authority, cap, pk_eph),
        "requirement 3: detectable, not merely fatal"))

    # A share aimed at a different ephemeral key must not assemble either.
    other_sk = gp.rand_scalar()
    out.append(_expect_refusal(
        "S1", "shares bound to another ephemeral key", "InvalidTag",
        lambda: tpre.unwrap(pr.reencrypt(q, cap, pk_eph, all_idx[:t]),
                            other_sk, cap)))
    return out


# --- S2: the combiner learns nothing, by ablation ----------------------------

def s2_combiner_ignorance(base: pr.Scenario) -> list[Check]:
    """Requirement 2, and the ablation that shows what discharges it.

    The check is structural, not cryptanalytic: no experiment can establish that
    a DDH instance is hard. What it can establish is that the combiner's view
    contains no representation of `K_e` and does not open without `sk_eph`, and
    that removing the blinding -- the single change -- makes both false. That is
    the same form of evidence as the flag ablation in Section `sec:results`.
    """
    out: list[Check] = []
    sc = _fresh(base)
    q, p = sc.quorum, sc.platform
    idx = [h.index for h in q.holders[:q.authority.t]]

    p.deny()
    req = p.request_recovery(pr.SafeEvidence("open-water", True, True))
    cap, pk_eph, sk_eph = req.capsule, p.pk_eph, p.sk_eph

    def combiner_view(re: tpre.ReEncrypted,
                      shares: Sequence[tpre.ReShare]) -> bytes:
        """Everything the combiner holds, serialised. It never sees `sk_eph`."""
        blob = gp.i2osp(re.z_masked) + gp.i2osp(re.e_masked)
        for s in shares:
            blob += b"".join(gp.i2osp(v) for v in
                             (s.d, s.e, s.a, s.b, s.c, s.z_s, s.z_k))
        return blob + cap.ct + cap.nonce + gp.i2osp(cap.c1)

    for blind, label in ((True, "as specified"), (False, "ablation: no blinding")):
        shares = [tpre.reshare(q.by_index[i].share, cap, pk_eph, blind=blind)
                  for i in idx]
        re = tpre.combine(shares, q.authority, cap, pk_eph)
        kek = tpre.unwrap(re, sk_eph, cap)
        view = combiner_view(re, shares)

        # The combiner has no sk_eph. Trying the identity exponent is exactly
        # what an unblinded assembly leaves it able to do.
        try:
            tpre.unwrap(re, 0, cap)
            opened = True
        except (InvalidTag, ValueError):
            opened = False

        out.append(Check(
            "S2", f"combiner cannot open the assembly ({label})",
            "cannot open", "cannot open" if not opened else "OPENS K_e",
            (not opened) if blind else opened,
            "the ablation is expected to open it; that is the finding"))
        out.append(Check(
            "S2", f"K_e does not appear in the combiner's view ({label})",
            "absent", "absent" if kek not in view else "present",
            kek not in view,
            "a byte-level check only; it cannot speak to computational hiding"))

    out.append(Check(
        "S2", "blinding is load-bearing", "yes",
        "yes" if all(c.passed for c in out) else "inconclusive",
        all(c.passed for c in out),
        "specified construction hides K_e from the combiner; the same code "
        "with k_i = 0 does not"))
    return out


# --- S3: the nine transcript bindings ---------------------------------------

def s3_transcript_binding(base: pr.Scenario) -> list[Check]:
    """Every field of `eq:token`, perturbed one at a time after signing.

    Section `sec:design` argues the transcript must be signed as a whole. The
    test of that argument is per-field, because a binding that covers eight of
    nine fields reads identically in prose and fails differently in practice.

    Two mechanisms catch the perturbations, and which one catches which is
    reported rather than hidden: seven fields are recomputed by the platform
    from its own state and fail the comparison; `counter` is not recomputed
    (the platform accepts any value above its own) and is caught by the
    signature; `pk_eph` fails the comparison too, since the platform knows which
    ephemeral key it generated.
    """
    out: list[Check] = []
    perturbations: dict[str, object] = {
        "vessel_id": b"IMO0000000",
        "h_puf": bytes(32),
        "nonce": b"\x01" * 16,
        "counter": None,      # filled in below: current + 1
        "h_safe": bytes(32),
        "h_att": bytes(32),
        "h_cust": bytes(32),
        "rc_digest": bytes(32),
        "pk_eph": None,       # filled in below: another group element
    }

    # Positive control first. Nine refusals mean nothing without it.
    sc = _fresh(base)
    ep = open_episode(sc)
    out.append(_expect_success(
        "S3", "control: unperturbed transcript is admitted",
        lambda: (sc.platform.admit(ep.token, ep.custody, ep.reencrypted)
                 or sc.platform.phase == pr.R0)))

    for fieldname in pr.TRANSCRIPT_FIELDS:
        sc = _fresh(base)
        ep = open_episode(sc)
        value = perturbations[fieldname]
        if fieldname == "counter":
            value = ep.transcript.counter + 1
        elif fieldname == "pk_eph":
            value = gp.exp(gp.G, gp.rand_scalar())
        tampered = pr.Token(replace(ep.transcript, **{fieldname: value}),
                            ep.token.signatures)
        expected = ("below_threshold" if fieldname == "counter"
                    else "transcript_mismatch")
        out.append(_expect_refusal(
            "S3", f"altering {fieldname} after signing", expected,
            lambda t=tampered, s=sc: s.platform.admit(t, ep.custody,
                                                      ep.reencrypted),
            "caught by the signature" if fieldname == "counter"
            else "caught by the platform's own recomputation"))
    return out


# --- S4: reuse and replay ----------------------------------------------------

def s4_reuse_and_replay(base: pr.Scenario) -> list[Check]:
    """The four lifetime rules of Section `sec:design`, and the one claim about
    them that is easy to state and easy to get wrong: that the nonce and the
    counter do different jobs.
    """
    out: list[Check] = []

    def spend(sc: pr.Scenario, ep: Episode) -> None:
        p = sc.platform
        p.admit(ep.token, ep.custody, ep.reencrypted)
        p.to_r1(); p.attest(); p.revalidate_sensing(); p.restore()
        p.resume_normal_operation()

    # 1. A consumed token re-presented, with its *original* attestation. The
    #    attestation is refused first: it names an episode that has closed.
    sc = _fresh(base)
    ep = open_episode(sc)
    p = sc.platform
    spend(sc, ep)
    p.deny()
    p.request_recovery(pr.SafeEvidence("open-water", True, True))
    out.append(_expect_refusal(
        "S4", "consumed token, original attestation", "custody_wrong_episode",
        lambda: p.admit(ep.token, ep.custody, ep.reencrypted),
        "A_cust carries the same nu as the token, so it is an independent "
        "replay barrier and it refuses before the token is examined"))

    # 1b. The same replay, but the adversary is granted a currently valid
    #     custody attestation. Now the token's own binding has to do the work.
    out.append(_expect_refusal(
        "S4", "consumed token, currently valid attestation",
        "transcript_mismatch",
        lambda: p.admit(ep.token, current_custody(sc), ep.reencrypted),
        "the new episode drew a new nonce and R3 committed a new epoch, so "
        "four of the nine fields differ"))

    # 2. The counter conjunct in isolation: a token freshly signed for the
    #    current episode in every respect except that it names a stale counter.
    sc = _fresh(base)
    ep = open_episode(sc)
    spend(sc, ep)
    p = sc.platform
    p.deny()
    stale = open_episode(sc, counter_override=p.counter)
    out.append(_expect_refusal(
        "S4", "freshly signed token naming a stale counter", "counter_not_fresh",
        lambda: p.admit(stale.token, stale.custody, stale.reencrypted),
        "everything else agrees, so this isolates the monotonic counter"))

    # 3. A token pre-minted during captivity for a future counter.
    sc = _fresh(base)
    ep_future = open_episode(sc, counter_override=sc.platform.counter + 5)
    p = sc.platform
    out.append(_expect_success(
        "S4", "control: a token for counter+5 is arithmetically fresh",
        lambda: ep_future.transcript.counter > p.counter,
        "the counter rule of Section sec:design is `exceeds`, not `equals`, "
        "so the counter does not catch this one"))
    p._abort_episode("adversary holds the platform; the episode lapses")
    p.request_recovery(pr.SafeEvidence("open-water", True, True))
    out.append(_expect_refusal(
        "S4", "pre-minted token spent in a later episode, valid attestation",
        "transcript_mismatch",
        lambda: p.admit(ep_future.token, current_custody(sc),
                        ep_future.reencrypted),
        "the nonce is what catches pre-minting, not the counter"))

    # 3. An abort does not consume the counter: retries are free.
    sc = _fresh(base)
    ep = open_episode(sc)
    p = sc.platform
    p.admit(ep.token, ep.custody, ep.reencrypted)
    p.to_r1()
    p.manifest = pr.Manifest({**p.baseline.entries, "autonomy-stack": bytes(32)})
    before = p.counter
    out.append(_expect_refusal(
        "S4", "abort at R1 on a tampered manifest", "attestation_mismatch",
        lambda: p.attest()))
    p.manifest = p.baseline  # off-board remediation
    out.append(_expect_success(
        "S4", "the counter was not consumed by the abort",
        lambda: p.counter == before and p.phase == pr.DENIED))
    ep2 = open_episode(sc)
    out.append(_expect_success(
        "S4", "retry at the same counter value succeeds",
        lambda: (p.admit(ep2.token, ep2.custody, ep2.reencrypted) is None
                 and p.to_r1() is None and p.attest() is None
                 and p.revalidate_sensing() is None
                 and p.restore() == sc.working_keys),
        "Section sec:staged: a class C adversary who forces retries must not "
        "thereby exhaust the owner's authority"))

    # 4. A capsule opening from one episode paired with a token from another
    #    must fail to decapsulate, not merely fail a later check.
    sc = _fresh(base)
    ep_a = open_episode(sc)
    p = sc.platform
    p._abort_episode("attempt A abandoned")
    ep_b = open_episode(sc)
    out.append(_expect_refusal(
        "S4", "episode B token with episode A capsule opening",
        "capsule_unwrap_failed",
        lambda: p.admit(ep_b.token, ep_b.custody, ep_a.reencrypted),
        "sk_eph for A was destroyed at the abort, so this is arithmetic, "
        "not policy"))
    return out


# --- S5: Table `tab:impl-reject` --------------------------------------------

def s5_rejection_suite(base: pr.Scenario) -> list[Check]:
    """The nine rows of Table `tab:impl-reject`, in order.

    Row 3 (a cloned board) and row 7 (below-threshold shares) are the two that
    need staging rather than mere perturbation, and both are staged out of the
    same scenario so that the only difference from the accepted path is the one
    under test.
    """
    out: list[Check] = []

    # Control.
    sc = _fresh(base)
    out.append(_expect_success(
        "S5", "control: the unperturbed exchange restores",
        lambda: pr.happy_path(sc) == sc.working_keys))

    # Row 1: a consumed authorisation re-presented, the adversary granted a
    # currently valid custody attestation so that the token has to be what
    # refuses. The table anticipated the counter; what actually refuses first is
    # the episode nonce, and the counter conjunct is exhibited separately in S4.
    sc = _fresh(base)
    ep = open_episode(sc)
    p = sc.platform
    p.admit(ep.token, ep.custody, ep.reencrypted)
    p.to_r1(); p.attest(); p.revalidate_sensing(); p.restore()
    p.resume_normal_operation(); p.deny()
    p.request_recovery(pr.SafeEvidence("open-water", True, True))
    out.append(_expect_refusal(
        "S5", "row 1: consumed token re-presented", "transcript_mismatch",
        lambda: p.admit(ep.token, current_custody(sc), ep.reencrypted),
        "P2. Refused on the episode nonce, not on the counter; see S4 for the "
        "counter conjunct in isolation"))

    # Row 2: a token from a different episode of the same denial.
    sc = _fresh(base)
    ep_a = open_episode(sc)
    p = sc.platform
    p._abort_episode("episode A abandoned")
    ep_b = open_episode(sc)
    out.append(_expect_refusal(
        "S5", "row 2: token from another episode", "transcript_mismatch",
        lambda: p.admit(ep_a.token, ep_b.custody, ep_b.reencrypted), "P2"))

    # Row 3: the same token presented on a cloned board.
    sc = _fresh(base)
    ep = open_episode(sc)
    clone = copy.deepcopy(sc.platform)
    clone.puf_response = os.urandom(32)  # a different device, same everything
    out.append(_expect_refusal(
        "S5", "row 3: token presented on a cloned board",
        "transcript_mismatch",
        lambda: clone.admit(ep.token, ep.custody, ep.reencrypted),
        "h_puf. The commitment differs, so the clone recomputes a different "
        "transcript"))

    # Row 4: stage bypass, R0 straight to R3.
    sc = _fresh(base)
    ep = open_episode(sc)
    p = sc.platform
    p.admit(ep.token, ep.custody, ep.reencrypted)
    out.append(_expect_refusal(
        "S5", "row 4: R0 to R3 directly", "stage_order",
        lambda: p.restore(), "P5"))

    # Row 5: a tampered manifest at R1.
    sc = _fresh(base)
    ep = open_episode(sc)
    p = sc.platform
    p.admit(ep.token, ep.custody, ep.reencrypted)
    p.to_r1()
    p.manifest = pr.Manifest({**p.baseline.entries,
                              "perception-model": bytes(32)})
    out.append(_expect_refusal(
        "S5", "row 5: tampered manifest at R1", "attestation_mismatch",
        lambda: p.attest(), "P8. Returns to denial rather than reporting"))

    # Row 6: no custody attestation.
    sc = _fresh(base)
    ep = open_episode(sc)
    out.append(_expect_refusal(
        "S5", "row 6: recovery without custody evidence",
        "no_custody_attestation",
        lambda: sc.platform.admit(ep.token, None, ep.reencrypted), "P3"))

    # Row 7: below-threshold shares. Two ways to be below t, and they are
    # refused by different mechanisms; both are reported.
    sc = _fresh(base)
    t = sc.quorum.authority.t
    idx = [h.index for h in sc.quorum.holders]
    ep = open_episode(sc, signers=idx[:t], reshare_signers=idx[:t - 1])
    out.append(_expect_refusal(
        "S5", "row 7a: t-1 re-encryption shares", "capsule_unwrap_failed",
        lambda: sc.platform.admit(ep.token, ep.custody, ep.reencrypted),
        "P10. The assembly does not open"))
    sc = _fresh(base)
    ep = open_episode(sc, signers=idx[:t - 1])
    out.append(_expect_refusal(
        "S5", "row 7b: t-1 signatures on the token", "below_threshold",
        lambda: sc.platform.admit(ep.token, ep.custody, ep.reencrypted),
        "P10. The token is not authority"))

    # Row 8: the situation degrades between stages.
    sc = _fresh(base)
    ep = open_episode(sc)
    p = sc.platform
    p.admit(ep.token, ep.custody, ep.reencrypted)
    p.to_r1(); p.attest()
    p.pending_safe = pr.SafeEvidence("narrow-channel", True, True)
    out.append(_expect_refusal(
        "S5", "row 8: safe state lost before R2 completes", "aborted_unsafe",
        lambda: p.revalidate_sensing(),
        "P4. Aborts to denial rather than holding a half-restored vessel"))

    # Row 9: absence of evidence is not evidence of safety.
    sc = _fresh(base)
    ep = open_episode(sc, kernel=None)
    out.append(_expect_refusal(
        "S5", "row 9: safe-state evidence absent on one source", "unsafe",
        lambda: sc.platform.admit(ep.token, ep.custody, ep.reencrypted),
        "P4. Section sec:safepred: none/safe reads false"))
    return out


# --- S6: where the time goes -------------------------------------------------

def _time(op: str, fn: Callable[[], object], n: int) -> Timing:
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return Timing(operation=op, n=n, mean_ms=statistics.fmean(samples),
                  median_ms=statistics.median(samples),
                  p95_ms=samples[min(len(samples) - 1, int(0.95 * len(samples)))],
                  max_ms=samples[-1])


def s6_latency_budget(base: pr.Scenario, n: int = 20) -> list[Timing]:
    """Decompose the restoration latency into its cryptographic parts.

    The point of this table is not that the numbers are small. It is that
    Section `sec:results` gives a mean restoration time of about 23 hours,
    dominated by quorum gathering at `rho = 0.25/h`, and these operations are
    milliseconds -- so no plausible improvement or degradation in them moves any
    quantitative claim the paper makes. Reading it the other way round: if these
    had come out in hours, Section `sec:results` would need redoing.

    The absolute values are a loose upper bound. Modular exponentiation here is
    CPython's `pow` on a 2048-bit modulus, which is far off a tuned library and
    much further off hardware. That looseness is in the safe direction for the
    argument being made.
    """
    sc = _fresh(base)
    q, p = sc.quorum, sc.platform
    idx = [h.index for h in q.holders[:q.authority.t]]
    p.deny()
    req = p.request_recovery(pr.SafeEvidence("open-water", True, True))
    cap, pk_eph, sk_eph = req.capsule, p.pk_eph, p.sk_eph
    cust = q.issuer.issue(vessel_id=p.vessel_id, counter=p.counter,
                          nonce=p.nonce, boarding_ref=b"b", issued_at=p.clock())
    transcript = replace(req.transcript, h_cust=cust.digest())
    one_share = tpre.reshare(q.by_index[idx[0]].share, cap, pk_eph)
    shares = [tpre.reshare(q.by_index[i].share, cap, pk_eph) for i in idx]
    re = tpre.combine(shares, q.authority, cap, pk_eph)
    tok = pr.issue_token(q, transcript, idx)
    msg = transcript.digest()

    out = [
        _time("capsule seal (platform, at R3)",
              lambda: tpre.seal(q.authority, os.urandom(32),
                                vessel_id=p.vessel_id, epoch=1), n),
        _time("re-encryption share + proof (one holder)",
              lambda: tpre.reshare(q.by_index[idx[0]].share, cap, pk_eph), n),
        _time("share proof verification (one share)",
              lambda: tpre.verify_reshare(one_share, q.authority, cap, pk_eph), n),
        _time(f"combine {q.authority.t} shares (incl. verification)",
              lambda: tpre.combine(shares, q.authority, cap, pk_eph), n),
        _time("capsule unwrap (platform)",
              lambda: tpre.unwrap(re, sk_eph, cap), n),
        _time("token signature (one holder)",
              lambda: pr.sign(q.holders[0].sk, msg), n),
        _time("token signature verification (one)",
              lambda: pr.verify(q.holders[0].pk, msg, tok.signatures[0][1]), n),
        _time("custody attestation issue",
              lambda: q.issuer.issue(vessel_id=p.vessel_id, counter=0,
                                     nonce=b"n" * 16, boarding_ref=b"b",
                                     issued_at=0), n),
        _time("transcript digest", lambda: transcript.digest(), n * 50),
        _time("manifest comparison (4 entries)",
              lambda: p.manifest.mismatches(p.baseline), n * 50),
        _time("ephemeral key generation",
              lambda: gp.exp(gp.G, gp.rand_scalar()), n),
    ]
    # The whole exchange, end to end, as the sum check.
    out.append(_time("full episode: denial through R3",
                     lambda: pr.happy_path(_fresh(base)), max(3, n // 4)))
    return out


# --- driver ------------------------------------------------------------------

EXPERIMENTS: dict[str, tuple[str, Callable[[pr.Scenario], list[Check]]]] = {
    "S1": ("Threshold assembly at t, refusal below t, attributed bad shares",
           s1_threshold_assembly),
    "S2": ("Combiner ignorance, by ablation against the unblinded construction",
           s2_combiner_ignorance),
    "S3": ("All nine transcript bindings of eq:token, one field at a time",
           s3_transcript_binding),
    "S4": ("Reuse, replay, pre-minting, and the cost-free retry",
           s4_reuse_and_replay),
    "S5": ("The nine rejection tests of Table tab:impl-reject", s5_rejection_suite),
}


@dataclass
class Results:
    checks: list[Check] = field(default_factory=list)
    timings: list[Timing] = field(default_factory=list)
    group_self_check: dict[str, bool] = field(default_factory=dict)
    parameters: dict[str, object] = field(default_factory=dict)
    #: What security hardware was found. Empty when the tier-1 experiments were
    #: not run; the reason is recorded here rather than left to be inferred from
    #: a missing section.
    hardware: dict[str, object] = field(default_factory=dict)

    @property
    def n_passed(self) -> int:
        return sum(c.passed for c in self.checks)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "group_self_check": self.group_self_check,
            "parameters": self.parameters,
            "hardware": self.hardware,
            "n_checks": len(self.checks),
            "n_passed": self.n_passed,
            "checks": [asdict(c) for c in self.checks],
            "timings": [asdict(t) for t in self.timings],
        }


def run_all(t: int = 3, orgs: Sequence[str] = pr.DEFAULT_ORGS,
            timing_n: int = 20, *, tier1: bool = True) -> Results:
    """Tier 0 always; tier 1 when security hardware is present.

    The tier-1 experiments are skipped rather than failed on a machine without a
    usable TPM, and the reason is recorded in `hardware`. A suite that fails on
    hardware it was never promised is a suite people learn to ignore.
    """
    base = pr.build_scenario(t=t, orgs=orgs)
    res = Results(group_self_check=gp.self_check(), parameters={
        "t": t, "n": len(orgs), "orgs": list(orgs),
        "group": "RFC 3526 2048-bit MODP, quadratic-residue subgroup",
        "group_bits": gp.P.bit_length(),
        "dem": "AES-256-GCM", "kdf": "HKDF-SHA256", "hash": "SHA-256",
        "signature": "Schnorr over the same group",
    })
    for _, fn in EXPERIMENTS.values():
        res.checks.extend(fn(base))
    res.timings = s6_latency_budget(base, n=timing_n)

    # Crash consistency of the capsule commit. Needs no hardware -- it is the
    # algorithm half of a claim whose other half is about the storage medium.
    from . import crash_experiments as cx
    res.checks.extend(cx.run_all())

    if tier1:
        # Imported here so that tier 0 runs on a machine with no TPM, and on a
        # platform where the Windows interfaces do not exist at all.
        from . import tpm_experiments as tx
        tpm_checks, status = tx.run_all(base)
        res.checks.extend(tpm_checks)
        res.hardware = status
    else:
        res.hardware = {"skipped": "tier 1 disabled by the caller"}
    return res
