"""The recovery exchange and the staged controller, as running code.

`modelcheck` checks the controller's *logic* over an abstract transition system
in which `auth` is one of four strings and `attested` is a Boolean. This module
checks the same controller where those Booleans have to be produced by real
signatures, a real transcript hash, a real monotonic counter, and a real
manifest comparison. The two are complementary and neither subsumes the other:
the abstract model can enumerate every reachable state and this one cannot; this
one can catch a binding that the abstract model assumes into existence.

Three things are worth stating about the correspondence.

**Same phases, same gates.** `NORMAL / DENIED / R0 / R1 / R2 / R3 / RESTORED` and
the conjuncts of each stage are the ones in `modelcheck.py`, deliberately so.
Where the abstract model writes `s.attested`, this module recomputes a manifest
digest and compares it to a baseline.

**The counter is `>`, not `= c+1`.** Section `sec:design` says the token's
counter value "exceeds every counter value the platform has accepted", and that
is what is implemented, rather than the tighter `= c + 1` that would also stop
pre-minting. The looser rule is the paper's, and it is implemented as written so
that experiment S4 can show which conjunct actually catches a pre-minted token.
It is the nonce, not the counter.

**`K_e` at R0, `K_w` at R3.** Section `sec:design` states both boundaries: the
module unwraps nothing under the ephemeral key until R0's conjuncts pass, and
neither output releases `K_w` before R3. Two distinct gates protect two distinct
things, and an abort at any stage zeroises both `sk_eph` and any recovered
`K_e`, since a token that survives an abort is exactly what Section
`sec:staged` says a class C adversary must not be able to manufacture.

**Where this instantiation is looser than the paper.** `Sig_{t-of-n}` is
realised as `t` individual Schnorr signatures from distinct registered holders
rather than as a threshold signature producing one constant-size token. The
verification predicate is the same; only the token size differs, and token size
is not something the paper claims anything about. A deployment would use a
threshold Schnorr scheme here.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Mapping, Sequence

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from . import group as gp
from . import tpre

DOMAIN_SIG = b"SGRD/v1/schnorr"
DOMAIN_TRANSCRIPT = b"SGRD/v1/transcript"
DOMAIN_CUSTODY = b"SGRD/v1/custody"

# --- phases, mirroring modelcheck.py ----------------------------------------

NORMAL, DENIED, RESTORED = "NORMAL", "DENIED", "RESTORED"
R0, R1, R2, R3 = "R0", "R1", "R2", "R3"
STAGES = (R0, R1, R2, R3)


class Refused(Exception):
    """The controller declined. Carries a machine-readable reason so that the
    rejection tests of Table `tab:impl-reject` can assert *why* a thing was
    refused and not merely that it was -- a test that passes for the wrong
    reason is worse than no test."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}{': ' + detail if detail else ''}")
        self.reason = reason
        self.detail = detail


def _h(domain: bytes, *parts: bytes) -> bytes:
    d = hashlib.sha256()
    d.update(len(domain).to_bytes(4, "big"))
    d.update(domain)
    for p in parts:
        d.update(len(p).to_bytes(4, "big"))
        d.update(p)
    return d.digest()


# --- Schnorr signatures ------------------------------------------------------

@dataclass(frozen=True)
class Signature:
    e: int
    s: int


def sign(sk: int, msg: bytes) -> Signature:
    k = gp.rand_scalar()
    r = gp.exp(gp.G, k)
    e = gp.hash_to_scalar(DOMAIN_SIG, r, gp.exp(gp.G, sk), msg)
    return Signature(e=e, s=(k + e * sk) % gp.Q)


def verify(pk: int, msg: bytes, sig: Signature) -> bool:
    if not gp.is_member(pk):
        return False
    r = gp.mul(gp.exp(gp.G, sig.s), gp.exp(gp.inv(pk), sig.e))
    return gp.hash_to_scalar(DOMAIN_SIG, r, pk, msg) == sig.e


# --- the transcript ----------------------------------------------------------
#
# Equation `eq:token`. The field order here IS the specification; changing it
# changes what a token means. The nine fields are kept as separate attributes
# rather than pre-hashed so that experiment S3 can perturb exactly one of them.

TRANSCRIPT_FIELDS = ("vessel_id", "h_puf", "nonce", "counter", "h_safe",
                     "h_att", "h_cust", "rc_digest", "pk_eph")


@dataclass(frozen=True)
class Transcript:
    """`T = H(id_S || h_puf || nu || c+1 || h_safe || h_att || h_cust || RC ||
    pk_eph)`, kept unhashed until :meth:`digest`.

    Section `sec:design` argues that this must be a transcript of the whole
    exchange and not a list of independently signed terms, because signing the
    terms separately leaves the token unbound to the key material it authorises.
    Experiment S3 tests the consequence one field at a time.
    """

    vessel_id: bytes
    h_puf: bytes
    nonce: bytes
    counter: int
    h_safe: bytes
    h_att: bytes
    h_cust: bytes
    rc_digest: bytes
    pk_eph: int

    def digest(self) -> bytes:
        return _h(DOMAIN_TRANSCRIPT, self.vessel_id, self.h_puf, self.nonce,
                  self.counter.to_bytes(8, "big"), self.h_safe, self.h_att,
                  self.h_cust, self.rc_digest, gp.i2osp(self.pk_eph))


# --- custody attestation -----------------------------------------------------

@dataclass(frozen=True)
class CustodyAttestation:
    """`A_cust = Sig_cust(id_S || c || nu || n_cust || t_iss || t_exp)`.

    The second authority of Section `sec:design`, and the only reason the
    platform can believe the `not Possessed` conjunct of R0. It carries the same
    `nu` and `c` the token binds to, so an attestation from one episode cannot
    be paired with a token from another.
    """

    vessel_id: bytes
    counter: int
    nonce: bytes
    boarding_ref: bytes  # n_cust: which boarding party, which report
    issued_at: int
    expires_at: int
    sig: Signature

    def message(self) -> bytes:
        return _h(DOMAIN_CUSTODY, self.vessel_id,
                  self.counter.to_bytes(8, "big"), self.nonce,
                  self.boarding_ref, self.issued_at.to_bytes(8, "big"),
                  self.expires_at.to_bytes(8, "big"))

    def digest(self) -> bytes:
        return _h(b"SGRD/v1/custody-digest", self.message(),
                  gp.i2osp(self.sig.e), gp.i2osp(self.sig.s))


@dataclass(frozen=True)
class CustodyIssuer:
    """Port or flag state. Section `sec:design` requires this issuer to hold no
    threshold share -- the two authorities are a second authority only if they
    are organisationally disjoint. :func:`check_disjoint` enforces it."""

    name: str
    sk: int
    pk: int

    def issue(self, *, vessel_id: bytes, counter: int, nonce: bytes,
              boarding_ref: bytes, issued_at: int,
              validity_s: int = 6 * 3600) -> CustodyAttestation:
        att = CustodyAttestation(
            vessel_id=vessel_id, counter=counter, nonce=nonce,
            boarding_ref=boarding_ref, issued_at=issued_at,
            expires_at=issued_at + validity_s, sig=Signature(0, 0))
        return replace(att, sig=sign(self.sk, att.message()))


def make_custody_issuer(name: str) -> CustodyIssuer:
    sk = gp.rand_scalar()
    return CustodyIssuer(name=name, sk=sk, pk=gp.exp(gp.G, sk))


# --- holders and the quorum --------------------------------------------------

@dataclass(frozen=True)
class Holder:
    """One of `Q_1..Q_n`: a threshold share of `K_r` plus a signing key."""

    index: int
    org: str
    share: tpre.HolderShare
    sk: int
    pk: int


@dataclass(frozen=True)
class Quorum:
    authority: tpre.RecoveryAuthority
    holders: tuple[Holder, ...]
    issuer: CustodyIssuer

    @property
    def by_index(self) -> Mapping[int, Holder]:
        return {h.index: h for h in self.holders}


def check_disjoint(q: Quorum) -> None:
    """The constraint Section `sec:design` states and no cryptography enforces.

    It is a registration-time assignment, so the only place it can be checked is
    a check like this one. Left implicit it is exactly the kind of constraint an
    operator discovers by violating it.
    """
    orgs = [h.org for h in q.holders]
    if len(set(orgs)) != len(orgs):
        raise ValueError(f"holders are not organisationally distinct: {orgs}")
    if q.issuer.name in orgs:
        raise ValueError(
            f"custody issuer {q.issuer.name!r} also holds a threshold share; "
            "the two authorities would not be disjoint")


def build_quorum(t: int, orgs: Sequence[str], issuer_name: str) -> Quorum:
    authority, shares = tpre.provision(t, len(orgs))
    holders = []
    for sh, org in zip(shares, orgs):
        sk = gp.rand_scalar()
        holders.append(Holder(index=sh.index, org=org, share=sh, sk=sk,
                              pk=gp.exp(gp.G, sk)))
    q = Quorum(authority=authority, holders=tuple(holders),
               issuer=make_custody_issuer(issuer_name))
    check_disjoint(q)
    return q


@dataclass(frozen=True)
class Token:
    """`tau`, plus the transcript it was computed over.

    The transcript travels with the token because the platform must recompute it
    from *its own* current values and compare -- that recomputation is the
    binding. A token carrying a transcript the platform does not independently
    arrive at is refused before any signature is checked.
    """

    transcript: Transcript
    signatures: tuple[tuple[int, Signature], ...]  # (holder index, sig)


def issue_token(q: Quorum, transcript: Transcript,
                signer_indices: Sequence[int]) -> Token:
    """The quorum signs. M2 of Algorithm `alg:proto` has each holder satisfy
    itself that custody has been regained before it does so; here that check is
    the caller's, since it is a human judgement and not a computation."""
    msg = transcript.digest()
    sigs = tuple((i, sign(q.by_index[i].sk, msg)) for i in signer_indices)
    return Token(transcript=transcript, signatures=sigs)


def reencrypt(q: Quorum, capsule: tpre.Capsule, pk_eph: int,
              signer_indices: Sequence[int], *,
              blind: bool = True) -> tpre.ReEncrypted:
    """The quorum's second output: the capsule re-encrypted under `pk_eph`."""
    shares = [tpre.reshare(q.by_index[i].share, capsule, pk_eph, blind=blind)
              for i in signer_indices]
    return tpre.combine(shares, q.authority, capsule, pk_eph)


# --- safe-state evidence -----------------------------------------------------
#
# Section `sec:safepred`. Two independent sources, and the disagreement table is
# fail-closed in every row that is not "both say safe". Absence is not safety:
# the row that reads "none / safe -> false" is the one that makes the predicate
# refuse rather than default.

SITUATIONS_ALLOWING_RECOVERY = frozenset(
    {"moored", "anchored", "open-water"})


@dataclass(frozen=True)
class SafeEvidence:
    """What the platform submits with a recovery request. `None` on either
    source means that source produced nothing, which is not the same as
    producing "safe"."""

    situation: str
    kernel_says_safe: bool | None
    roc_says_safe: bool | None

    def holds(self) -> bool:
        if self.situation not in SITUATIONS_ALLOWING_RECOVERY:
            return False  # narrow waters, heavy weather: H1 and H4 dominate
        return self.kernel_says_safe is True and self.roc_says_safe is True

    def digest(self) -> bytes:
        def enc(v: bool | None) -> bytes:
            return b"?" if v is None else (b"1" if v else b"0")
        return _h(b"SGRD/v1/safe", self.situation.encode(),
                  enc(self.kernel_says_safe), enc(self.roc_says_safe))


# --- measured state ----------------------------------------------------------

@dataclass(frozen=True)
class Manifest:
    """The measured configuration: component name to digest.

    A real platform would take these from a measured boot chain. Here they are
    digests of the artefacts themselves, which exercises the comparison logic
    without claiming anything about how the measurements are rooted -- Section
    `sec:impl` is explicit that a hardware root of trust is not demonstrated.
    """

    entries: Mapping[str, bytes]

    def digest(self) -> bytes:
        return _h(b"SGRD/v1/manifest",
                  *[k.encode() + b"=" + v for k, v in sorted(self.entries.items())])

    def mismatches(self, baseline: "Manifest") -> list[str]:
        keys = set(self.entries) | set(baseline.entries)
        return sorted(k for k in keys
                      if self.entries.get(k) != baseline.entries.get(k))


# --- the platform ------------------------------------------------------------

@dataclass
class RecoveryRequest:
    """What the platform emits when it asks for help. It commits to its own
    measured state and navigational situation *before* the quorum issues
    authority, which is the ordering property Section `sec:proto` calls
    essential."""

    transcript: Transcript
    capsule: tpre.Capsule
    safe: SafeEvidence
    manifest: Manifest


@dataclass
class Platform:
    """One vessel's module boundary, as a state machine.

    Every field that would live inside a cryptographic module is here; nothing
    that would live outside it is. `sk_eph` and `kek` are set to `None` rather
    than to a stale value when they are destroyed, so a use-after-zeroise raises
    instead of silently working.
    """

    vessel_id: bytes
    authority: tpre.RecoveryAuthority
    issuer_pk: int
    puf_response: bytes  # the device secret; never leaves, never hashed out
    baseline: Manifest
    manifest: Manifest

    kek: bytes | None = None  # K_e
    wrapped_keys: tuple[bytes, bytes] | None = None  # (nonce, W) = Wrap_{K_e}(K_w)
    working_keys: bytes | None = None  # K_w, present only outside denial

    capsule: tpre.Capsule | None = None
    counter: int = 0
    epoch: int = 0

    phase: str = NORMAL
    denied: bool = False
    possessed: bool = False
    sensor_trusted: bool = True
    command_owner: str = "roc"

    # Per-episode ephemeral state.
    nonce: bytes | None = None
    sk_eph: int | None = None
    pk_eph: int | None = None
    pending_safe: SafeEvidence | None = None
    admitted_token: Token | None = None
    recovered_kek: bytes | None = None

    # The holders' signing keys, as recorded at registration. Kept as a plain
    # mapping so a test can hand the platform a different registry -- which is
    # how the cloned-board and wrong-quorum rejections are staged.
    holder_pks: dict[int, int] = field(default_factory=dict)

    clock: Callable[[], int] = field(default=lambda: 1_700_000_000)
    log: list[str] = field(default_factory=list)

    # -- device commitment ---------------------------------------------------

    @property
    def h_puf(self) -> bytes:
        """`h_puf`: a commitment to the device's challenge-response behaviour.

        Section `sec:design` insists this is a commitment and not a key, and
        that the quorum never learns the response. Hashing it here, and never
        transmitting `puf_response`, is that distinction made concrete: a cloned
        board with a different response produces a different commitment and
        therefore recomputes a different transcript.
        """
        return _h(b"SGRD/v1/puf", self.vessel_id, self.puf_response)

    # -- provisioning and normal operation -----------------------------------

    def provision(self, working_keys: bytes) -> None:
        """Draw `K_e`, wrap `K_w` under it, and commit the first capsule."""
        self.kek = os.urandom(32)
        self.working_keys = working_keys
        self._rewrap()
        self._commit_capsule()

    def _rewrap(self) -> None:
        assert self.kek is not None and self.working_keys is not None
        nonce = os.urandom(12)
        ct = AESGCM(self.kek).encrypt(nonce, self.working_keys, self.vessel_id)
        self.wrapped_keys = (nonce, ct)

    def _commit_capsule(self) -> None:
        """Commit the capsule for the current epoch, then and only then release
        the previous one. Section `sec:design` requires this order and requires
        it to be atomic; the atomicity across power loss is experiment S7's
        subject and is not tested here."""
        assert self.kek is not None
        self.epoch += 1
        fresh = tpre.seal(self.authority, self.kek,
                          vessel_id=self.vessel_id, epoch=self.epoch)
        _ = fresh.digest()  # the "digest confirmed" step of Section sec:design
        self.capsule = fresh

    # -- denial --------------------------------------------------------------

    def deny(self) -> None:
        """Zeroise `K_e` and the working memory; keep the committed capsule.

        `wrapped_keys` and the encrypted payload stay on external storage, which
        is the point: without `K_e` they are inert, and the plaintext never left
        the memory-encrypted domain.
        """
        if self.capsule is None:
            raise Refused("not_provisioned")
        self.kek = None
        self.working_keys = None
        self.phase = DENIED
        self.denied = True
        self.sensor_trusted = False
        self.command_owner = "autonomous"
        self._abort_episode("denial fired")
        self.log.append(f"deny epoch={self.epoch} counter={self.counter}")

    def _abort_episode(self, why: str) -> None:
        """Destroy per-episode material. Called on abort and on completion.

        Section `sec:design`: `sk_eph` is generated for one `nu` and is stored
        nowhere else, so a capsule opening encapsulated under another session's
        key fails to decapsulate at all rather than failing a later check.
        """
        self.nonce = None
        self.sk_eph = None
        self.pk_eph = None
        self.pending_safe = None
        self.admitted_token = None
        self.recovered_kek = None
        if why:
            self.log.append(f"abort: {why}")

    # -- the request ---------------------------------------------------------

    def request_recovery(self, safe: SafeEvidence) -> RecoveryRequest:
        """Draw a fresh nonce and ephemeral key pair and commit to the current
        measured state and navigational situation."""
        if not self.denied:
            raise Refused("not_denied")
        if self.capsule is None:
            raise Refused("no_capsule")
        self.nonce = os.urandom(16)
        self.sk_eph = gp.rand_scalar()
        self.pk_eph = gp.exp(gp.G, self.sk_eph)
        self.pending_safe = safe
        transcript = Transcript(
            vessel_id=self.vessel_id, h_puf=self.h_puf, nonce=self.nonce,
            counter=self.counter + 1, h_safe=safe.digest(),
            h_att=self.manifest.digest(), h_cust=b"", # filled by the caller
            rc_digest=self.capsule.digest(), pk_eph=self.pk_eph)
        return RecoveryRequest(transcript=transcript, capsule=self.capsule,
                               safe=safe, manifest=self.manifest)

    # -- R0: admission -------------------------------------------------------

    def admit(self, token: Token, custody: CustodyAttestation | None,
              reencrypted: tpre.ReEncrypted) -> None:
        """The R0 gate: `Auth AND Safe AND NOT Possessed AND NOT Used`.

        The order of checks below is not the order of importance; it is chosen
        so that each rejection test in Table `tab:impl-reject` trips a distinct,
        named reason. A gate that refuses everything for the same reason has not
        been tested.
        """
        if self.phase != DENIED:
            raise Refused("wrong_phase", f"in {self.phase}")
        if self.nonce is None or self.pk_eph is None:
            raise Refused("no_outstanding_request")

        # --- custody: the second, disjoint authority -------------------------
        if custody is None:
            raise Refused("no_custody_attestation")
        h_cust = custody.digest()
        if not verify(self.issuer_pk, custody.message(), custody.sig):
            raise Refused("custody_signature_invalid")
        if custody.vessel_id != self.vessel_id:
            raise Refused("custody_wrong_vessel")
        if custody.nonce != self.nonce:
            raise Refused("custody_wrong_episode", "nonce mismatch")
        if custody.counter != self.counter:
            raise Refused("custody_wrong_epoch", "counter mismatch")
        now = self.clock()
        if not custody.issued_at <= now < custody.expires_at:
            raise Refused("custody_expired")

        # --- the transcript, recomputed from the platform's own values -------
        # This single comparison discharges all nine bindings of eq:token. The
        # per-field experiment S3 exists because "one comparison covers nine
        # fields" is a claim about the encoding, and claims about encodings are
        # where binding bugs live.
        expected = Transcript(
            vessel_id=self.vessel_id, h_puf=self.h_puf, nonce=self.nonce,
            counter=token.transcript.counter, h_safe=self.pending_safe.digest(),
            h_att=self.manifest.digest(), h_cust=h_cust,
            rc_digest=self.capsule.digest(), pk_eph=self.pk_eph)
        if token.transcript != expected:
            differing = [f for f in TRANSCRIPT_FIELDS
                         if getattr(token.transcript, f) != getattr(expected, f)]
            raise Refused("transcript_mismatch", ",".join(differing))

        # --- freshness and single use ---------------------------------------
        if token.transcript.counter <= self.counter:
            raise Refused("counter_not_fresh",
                          f"{token.transcript.counter} <= {self.counter}")

        # --- the quorum signed it -------------------------------------------
        msg = token.transcript.digest()
        seen: set[int] = set()
        good = 0
        for idx, sig in token.signatures:
            vk = self.authority.verification_keys.get(idx)
            if vk is None or idx in seen:
                continue
            seen.add(idx)
            # The signing key is separate from the threshold share; the platform
            # knows the signing keys from registration. Modelled by deriving the
            # verification key from the registry the platform was provisioned
            # with, held on the quorum object in this implementation.
            if verify(self._holder_pk(idx), msg, sig):
                good += 1
        if good < self.authority.t:
            raise Refused("below_threshold", f"{good} of {self.authority.t}")

        # --- the safe-state envelope ----------------------------------------
        if not self.pending_safe.holds():
            raise Refused("unsafe", self.pending_safe.situation)
        if self.possessed:
            raise Refused("still_possessed")

        # --- only now may anything be decapsulated ---------------------------
        try:
            self.recovered_kek = tpre.unwrap(reencrypted, self.sk_eph,
                                             self.capsule)
        except (InvalidTag, ValueError) as exc:
            raise Refused("capsule_unwrap_failed", type(exc).__name__) from exc

        self.admitted_token = token
        self.phase = R0
        self.log.append(f"R0 admitted counter={token.transcript.counter}")

    def _holder_pk(self, idx: int) -> int:
        return self.holder_pks.get(idx, 0)

    # -- R1..R3 --------------------------------------------------------------

    def _gate(self) -> None:
        """The conjuncts every stage re-evaluates. Section `sec:staged` requires
        the safe-state predicate to be re-evaluated at each stage rather than
        once at admission, because the situation moves while the stages run."""
        if self.possessed:
            self.phase = DENIED
            self._abort_episode("possession regained by the adversary")
            raise Refused("aborted_possessed")
        if self.pending_safe is None or not self.pending_safe.holds():
            self.phase = DENIED
            self._abort_episode("safe-state predicate no longer holds")
            raise Refused("aborted_unsafe")

    def to_r1(self) -> None:
        if self.phase != R0:
            raise Refused("stage_order", f"R1 from {self.phase}")
        self._gate()
        self.phase = R1

    def attest(self) -> None:
        """R1: re-attestation. Nothing from the protected payload executes here.

        A mismatch is not an error to be reported and worked around; it returns
        the platform to denial. Section `sec:staged` hazard 4.
        """
        if self.phase != R1:
            raise Refused("stage_order", f"attest from {self.phase}")
        self._gate()
        bad = self.manifest.mismatches(self.baseline)
        if bad:
            self.phase = DENIED
            self._abort_episode(f"attestation mismatch: {','.join(bad)}")
            raise Refused("attestation_mismatch", ",".join(bad))
        self.phase = R2

    def revalidate_sensing(self) -> None:
        """R2: supervision re-established, sensing and positioning revalidated,
        command returned from the boarding party to the ROC."""
        if self.phase != R2:
            raise Refused("stage_order", f"R2 from {self.phase}")
        self._gate()
        self.sensor_trusted = True
        self.command_owner = "roc"
        self.phase = R3

    def restore(self) -> bytes:
        """R3: unwrap `K_w`, advance the counter, re-arm with a fresh `K_e`.

        This is the only place the counter is consumed, which is what makes an
        abort at R0-R2 free: a class C adversary who forces retries does not
        thereby exhaust the owner's authority.
        """
        if self.phase != R3:
            raise Refused("stage_order", f"R3 from {self.phase}")
        self._gate()
        if not self.sensor_trusted:
            raise Refused("sensing_not_revalidated")
        if self.recovered_kek is None or self.wrapped_keys is None:
            raise Refused("nothing_to_unwrap")

        nonce, ct = self.wrapped_keys
        try:
            kw = AESGCM(self.recovered_kek).decrypt(nonce, ct, self.vessel_id)
        except InvalidTag as exc:
            raise Refused("working_key_unwrap_failed") from exc

        self.counter = self.admitted_token.transcript.counter
        self.working_keys = kw
        # Re-arm: a fresh K_e and a fresh capsule, committed before the old
        # epoch's material is released.
        self.kek = os.urandom(32)
        self._rewrap()
        self._commit_capsule()
        self.phase = RESTORED
        self.denied = False
        self._abort_episode("")
        self.log.append(f"R3 restored counter={self.counter} epoch={self.epoch}")
        return kw

    def resume_normal_operation(self) -> None:
        if self.phase != RESTORED:
            raise Refused("stage_order", f"resume from {self.phase}")
        self.phase = NORMAL


# --- assembling a whole scenario --------------------------------------------

DEFAULT_ORGS = ("operator-roc", "class-society", "owner-escrow",
                "commercial-escrow")


@dataclass
class Scenario:
    """One provisioned vessel plus its quorum, ready to be driven.

    Building this is the expensive part -- provisioning is `n + 1` scalar
    multiplications and the capsule another two -- so the experiments build one
    and reuse it rather than one per test.
    """

    quorum: Quorum
    platform: Platform
    working_keys: bytes


def build_scenario(t: int = 3, orgs: Sequence[str] = DEFAULT_ORGS,
                   issuer_name: str = "port-state",
                   vessel_id: bytes = b"IMO9074729") -> Scenario:
    q = build_quorum(t, list(orgs), issuer_name)
    baseline = Manifest({
        "bootloader": _h(b"art", b"bootloader-v1"),
        "kernel": _h(b"art", b"nav-kernel-v3"),
        "autonomy-stack": _h(b"art", b"cpi-stack-v7"),
        "perception-model": _h(b"art", b"model-2026-02"),
    })
    kw = os.urandom(64)
    p = Platform(
        vessel_id=vessel_id, authority=q.authority, issuer_pk=q.issuer.pk,
        puf_response=os.urandom(32), baseline=baseline, manifest=baseline,
        holder_pks={h.index: h.pk for h in q.holders})
    p.provision(kw)
    return Scenario(quorum=q, platform=p, working_keys=kw)


def happy_path(sc: Scenario, *, signers: Iterable[int] | None = None,
               situation: str = "open-water") -> bytes:
    """Denial through R3, with nothing perturbed. The positive control every
    rejection test needs: a suite of nine refusals proves nothing unless the
    unperturbed exchange is known to succeed."""
    p = sc.platform
    if not p.denied:
        p.deny()
    idx = list(signers) if signers is not None else \
        [h.index for h in sc.quorum.holders[:sc.quorum.authority.t]]
    req = p.request_recovery(SafeEvidence(situation, True, True))
    cust = sc.quorum.issuer.issue(
        vessel_id=p.vessel_id, counter=p.counter, nonce=p.nonce,
        boarding_ref=b"boarding-2026-014", issued_at=p.clock())
    transcript = replace(req.transcript, h_cust=cust.digest())
    token = issue_token(sc.quorum, transcript, idx)
    re = reencrypt(sc.quorum, req.capsule, p.pk_eph, idx)
    p.admit(token, cust, re)
    p.to_r1()
    p.attest()
    p.revalidate_sensing()
    kw = p.restore()
    p.resume_normal_operation()
    return kw
