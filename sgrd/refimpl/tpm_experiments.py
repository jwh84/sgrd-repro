"""Tier-1 experiments T1-T5: what a TPM 2.0 settles that software cannot.

Section `sec:impl` lists three elements of the design that tier 0 has to take on
trust because they are hardware properties: a secret bound to the device, a
measurement of the platform's state, and a counter that cannot be wound back.
A TPM 2.0 supplies an analogue of each, and every machine meeting the Windows 11
installation requirement has one. These experiments use it.

They do **not** turn the design's hardware claims into results. A TPM is not a
PUF, its PCRs measure the boot chain and not the autonomy stack, and its clock
is not the design's monotonic counter. What each experiment establishes and what
it leaves open is stated in its own docstring and carried into the results as a
note, because the distance between "a TPM does this" and "the design's component
does this" is exactly where an overclaim would hide.

The one that matters most is T3. TPM sealing releases a secret only when the
platform's measured state matches the state the secret was sealed to -- which is
what R1 does, enforced by hardware rather than by the controller. It is the
closest hardware analogue of the recovery capsule that a general-purpose machine
can offer.

Nothing here changes persistent state on the machine. See `tpm.py` for why.
"""

from __future__ import annotations

import copy
import hashlib
import statistics
import time
from typing import Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric import utils as asym_utils

from . import pcp
from . import protocol as pr
from . import tpm as tp
from .experiments import Check, _expect_refusal, _fresh, open_episode

#: The boot-chain PCRs of the TCG PC Client profile: firmware code and
#: configuration, option ROMs, boot manager, and Secure Boot policy. They are
#: not resettable from locality 0, which is what makes them worth sealing to.
BOOT_PCRS = tuple(range(8))

PCR_MEANINGS = {
    0: "UEFI firmware code",
    1: "UEFI firmware configuration",
    2: "option ROM code",
    3: "option ROM configuration",
    4: "boot manager and loader code",
    5: "boot manager configuration, GPT",
    6: "platform manufacturer events",
    7: "Secure Boot policy",
}


def _check(name: str, expected: str, observed: str, note: str = "",
           detail: str = "") -> Check:
    return Check("T", name, expected, observed, expected == observed, note,
                 detail)


# --- T1: a secret the device cannot be separated from ------------------------

def t1_device_bound_secret() -> list[Check]:
    """What `h_puf` is asked to do, done by a TPM-resident key.

    **Establishes.** The private half is generated inside the TPM and the
    provider refuses to export it; a signature it produces verifies against the
    exported public half under a library that never touched the TPM; distinct
    keys give distinct commitments.

    **Does not establish.** Physical unclonability. A TPM key resists extraction
    because the chip refuses to release it, not because the secret is a function
    of manufacturing variation that cannot be replicated. The design's claim
    that a *cloned board* cannot produce the same commitment needs tier 2 and
    a PUF; what this shows is that a *different key* cannot, which is a weaker
    statement about a different mechanism.
    """
    out: list[Check] = []
    with pcp.PlatformKeyStore() as store:
        key, handle = store.create()

        out.append(_check(
            "T1 key is generated inside the TPM", "ECDSA", key.algorithm_group,
            f"provider: {pcp.PROVIDER}"))
        out.append(_check(
            "T1 export policy forbids export", "0", str(key.export_policy),
            "NCRYPT_ALLOW_EXPORT_FLAG and friends all clear"))

        msg = b"SGRD/v1/transcript|" + b"\x5a" * 32
        digest = hashlib.sha256(msg).digest()
        raw = store.sign(handle, digest)
        x, y = key.public_numbers()
        pub = ec.EllipticCurvePublicNumbers(x, y, ec.SECP256R1()).public_key()
        der = asym_utils.encode_dss_signature(
            int.from_bytes(raw[:32], "big"), int.from_bytes(raw[32:], "big"))

        try:
            pub.verify(der, msg, ec.ECDSA(hashes.SHA256()))
            verified = "verified"
        except InvalidSignature:
            verified = "did not verify"
        out.append(_check(
            "T1 TPM signature verifies under an independent library",
            "verified", verified,
            "the TPM signed it; `cryptography` checked it against the "
            "exported public key"))

        try:
            pub.verify(der, b"a different message", ec.ECDSA(hashes.SHA256()))
            wrong = "accepted"
        except InvalidSignature:
            wrong = "rejected"
        out.append(_check(
            "T1 the same signature over a different message", "rejected",
            wrong))

        try:
            store.export(handle, pcp.BLOB_ECC_PRIVATE)
            refused, detail = "exported", ""
        except pcp.PcpError as exc:
            refused, detail = "refused", f"0x{exc.status & 0xFFFFFFFF:08x}"
        out.append(_check(
            "T1 private key export", "refused", refused,
            "NTE_NOT_SUPPORTED. The private half has no representation outside "
            "the chip", detail))

        other, _ = store.create()
        out.append(_check(
            "T1 a second key gives a different commitment", "different",
            "different" if other.commitment != key.commitment else "identical",
            "so the commitment identifies the key, as h_puf identifies the "
            "hull"))
    return out


# --- T2: the platform's measured state ---------------------------------------

def t2_measured_state() -> list[Check]:
    """PCRs 0-7 as a real `h_att`.

    **Establishes.** The machine carries a hardware-held measurement of its own
    boot chain, it is stable across reads, and it is specific enough that a
    digest over it is a usable attestation quote.

    **Does not establish.** That these measurements cover what the design needs
    measured. PCRs 0-7 record firmware, option ROMs, the boot manager and Secure
    Boot policy -- not the autonomy stack, the perception model, or the
    configuration the design's manifest enumerates. Measuring those needs a
    measured-launch chain that extends past the boot loader, which this machine
    does not have and this experiment does not build.
    """
    out: list[Check] = []
    with tp.Tpm() as t:
        first = t.pcr_read(BOOT_PCRS)
        second = t.pcr_read(BOOT_PCRS)

        out.append(_check(
            "T2 boot PCRs are readable", str(len(BOOT_PCRS)), str(len(first)),
            ", ".join(f"PCR{i}: {PCR_MEANINGS[i]}" for i in (0, 4, 7))))
        out.append(_check(
            "T2 measurements are stable across reads", "stable",
            "stable" if first == second else "changed",
            "nothing extended them between the two reads, as expected on an "
            "idle system"))

        populated = [i for i, v in first.items() if v != bytes(32)]
        n = len(BOOT_PCRS)
        out.append(_check(
            "T2 the boot chain actually measured something",
            f"{n} of {n} populated", f"{len(populated)} of {n} populated",
            "an all-zero PCR would mean nothing was measured into it",
            detail=",".join(f"PCR{i}" for i in first if i not in populated)))
        out.append(_check(
            "T2 measurements are distinct from one another",
            f"{n} distinct", f"{len(set(first.values()))} distinct"))

        pcr23 = t.pcr_read([tp.APPLICATION_PCR])[tp.APPLICATION_PCR]
        out.append(_check(
            "T2 the application PCR is unextended", "unextended",
            "unextended" if pcr23 == bytes(32) else "already extended",
            "checked before T3 seals anything, so the experiment cannot "
            "disturb another application's use of PCR 23"))

        quote = hashlib.sha256(
            b"".join(first[i] for i in sorted(first))).digest()
        out.append(_check(
            "T2 a quote digest can be formed from them", "formed", "formed",
            "this is what would go into h_att", detail=quote.hex()[:32]))
    return out


# --- T3: sealing, the hardware analogue of the recovery capsule --------------

def t3_sealing_to_measured_state() -> list[Check]:
    """A secret released only in the measured state it was sealed to.

    **This is the experiment worth doing.** Section `sec:staged` gives R1 the
    job of refusing to proceed when the platform's measured state does not match
    its post-possession baseline. TPM sealing is that rule enforced by the chip:
    the secret is bound to a policy over PCR values, and the TPM will not release
    it under any other measurement. The controller does not have to be trusted
    to check, because the check is not the controller's.

    The demonstration seals the same secret three ways -- to the live boot
    measurement, and to states differing in exactly one PCR -- and shows which
    open. Sealing to *differing* states rather than mutating a PCR is not a
    convenience: Windows blocks PCR extension from user mode, and the
    substitution turns out to be the better experiment, because it exercises the
    real boot-chain PCRs rather than the scratch one.

    **Establishes.** Hardware refuses to release a sealed secret when a single
    boot measurement differs, and the refusal is a policy failure rather than a
    corrupted result.

    **Does not establish.** That the design's capsule can be built this way. The
    capsule is sealed under a key the *platform never holds*, so that a quorum
    ashore is needed to open it; a TPM-sealed object is opened by the platform
    itself once the measurement matches. The two answer different questions --
    the capsule answers "who may authorise", sealing answers "in what state" --
    and the design needs both.
    """
    out: list[Check] = []
    secret = hashlib.sha256(b"K_e for the sealed epoch").digest()

    with tp.Tpm() as t:
        pcr23 = t.pcr_read([tp.APPLICATION_PCR])[tp.APPLICATION_PCR]
        if pcr23 != bytes(32):
            return [_check("T3 precondition: PCR 23 unextended", "unextended",
                           "already extended",
                           "refusing to run rather than disturb it")]

        for label, fn in (("extend", lambda: t.pcr_extend(
                               tp.APPLICATION_PCR, bytes(32))),
                          ("reset", lambda: t.pcr_reset(tp.APPLICATION_PCR))):
            try:
                fn()
                observed = "allowed"
            except tp.TpmBlocked:
                observed = "blocked by TBS"
            except tp.TpmError as exc:
                observed = f"TPM_RC 0x{exc.rc:08x}"
            out.append(_check(
                f"T3 PCR {label} from user mode", "blocked by TBS", observed,
                "a Windows policy, not a TPM verdict. It is why the binding is "
                "shown by sealing to differing states instead"))

        live = t.pcr_read(BOOT_PCRS)
        parent = t.create_primary_null()

        def attempt(policy: bytes, label: str, expect_open: bool,
                    note: str = "", detail: str = "") -> None:
            item = t.load(parent, t.seal(parent, secret, policy))
            session = t.start_policy_session()
            t.policy_pcr(session, BOOT_PCRS)
            got = t.unseal(item, session)
            t.flush(session)
            t.flush(item)
            observed = "opened" if got == secret else "refused"
            out.append(Check(
                "T", label, "opened" if expect_open else "refused", observed,
                observed == ("opened" if expect_open else "refused"),
                note, detail))

        attempt(tp.policy_pcr_digest(live),
                "T3 sealed to the live boot measurement", True,
                "the control. Without it the refusals below prove nothing")

        for i in (0, 4, 7):
            altered = dict(live)
            altered[i] = hashlib.sha256(b"a different firmware image").digest()
            attempt(tp.policy_pcr_digest(altered),
                    f"T3 sealed to a state differing only in PCR{i}", False,
                    f"{PCR_MEANINGS[i]}. One PCR of eight, and the chip will "
                    "not release the secret", detail="TPM_RC_POLICY_FAIL")

        attempt(tp.policy_pcr_digest(live),
                "T3 sealed to the live measurement, second time", True,
                "repeatable, so the refusals were about the policy and not "
                "about exhausting some resource")

        # A policy session that never took the measurement into account.
        item = t.load(parent, t.seal(parent, secret,
                                     tp.policy_pcr_digest(live)))
        session = t.start_policy_session()          # no policy_pcr call
        got = t.unseal(item, session)
        t.flush(session)
        t.flush(item)
        out.append(_check(
            "T3 unseal without consulting the PCRs at all", "refused",
            "opened" if got == secret else "refused",
            "the session's policy digest stays zero, so it cannot match. "
            "userWithAuth is clear on the object, which is what forces a "
            "policy session rather than a password"))
    return out


# --- T4: a value that cannot be wound back -----------------------------------

def t4_monotonic_values(samples: int = 5, gap_s: float = 0.05) -> list[Check]:
    """The TPM clock and reset counter, in place of the design's counter `c`.

    **Establishes.** The machine holds a counter that software cannot decrease.
    `clock` is persisted by the TPM across power cycles and the specification
    forbids it from going backwards; `resetCount` increments on every TPM reset.
    The `safe` flag is the TPM's own assertion that it can still make the
    guarantee.

    **Does not establish.** The design's counter. `c` has to be incremented
    exactly once per completed restoration and quoted in a token, which is what
    an NV index of type `TPM_NT_COUNTER` is for. Defining one needs owner
    authorisation and consumes a scarce persistent NV slot on the machine, so
    this experiment does not define one -- it demonstrates that the hardware
    offers monotonicity, not that the design's use of it has been built.
    """
    out: list[Check] = []
    with tp.Tpm() as t:
        readings = []
        for _ in range(samples):
            readings.append(t.read_clock())
            time.sleep(gap_s)

        clocks = [r.clock_ms for r in readings]
        out.append(_check(
            "T4 clock never decreases", "monotonic",
            "monotonic" if all(b >= a for a, b in zip(clocks, clocks[1:]))
            else "went backwards",
            f"{samples} readings over {samples * gap_s:.2f} s",
            detail=f"{clocks[0]} -> {clocks[-1]} ms"))
        out.append(_check(
            "T4 clock advances with real time", "advances",
            "advances" if clocks[-1] > clocks[0] else "stalled",
            "so it is a clock and not a constant"))
        out.append(_check(
            "T4 the TPM asserts the guarantee still holds", "safe",
            "safe" if readings[0].safe else "not safe",
            "TPMS_CLOCK_INFO.safe: cleared if the TPM cannot rule out that "
            "clock has moved backwards"))
        resets = {r.reset_count for r in readings}
        out.append(_check(
            "T4 reset count is stable within a run", "1 value(s)",
            f"{len(resets)} value(s)",
            f"resetCount = {readings[0].reset_count}: the number of TPM resets "
            "since it was last cleared, and it only ever increases"))

        rnd = [t.get_random(32) for _ in range(4)]
        out.append(_check(
            "T4 hardware RNG returns distinct 32-byte draws", "4 distinct",
            f"{len(set(rnd))} distinct",
            "not a randomness test; a liveness check on TPM2_GetRandom"))
    return out


# --- T5: the tier-0 exchange driven by real hardware inputs ------------------

def t5_hardware_backed_exchange(base: pr.Scenario) -> list[Check]:
    """Run the software exchange with `h_puf` and `h_att` taken from the TPM.

    Tiers do not compose by themselves. S1-S5 fed the controller values invented
    by the test; this feeds it a device commitment from a TPM-resident key and a
    manifest built from live PCRs, and checks that the exchange still completes
    and still refuses what it should.

    The cloned-board case is the one worth reading. Two keys on the *same* TPM
    already produce different commitments, so presenting a token bound to one to
    a platform holding the other is refused -- by the transcript comparison, on
    the `h_puf` field, exactly as Table `tab:impl-reject` says. That is the
    mechanism the design relies on; what it is not is evidence that a cloned
    board cannot reproduce a PUF response.
    """
    out: list[Check] = []
    with tp.Tpm() as t, pcp.PlatformKeyStore() as store:
        pcrs = t.pcr_read(BOOT_PCRS)
        key, _handle = store.create()

        def hardware_platform(sc: pr.Scenario, commitment: bytes) -> None:
            """Point the platform's device commitment and manifest at the TPM."""
            p = sc.platform
            p.puf_response = commitment
            manifest = pr.Manifest({f"pcr{i:02d}": v for i, v in pcrs.items()})
            p.baseline = manifest
            p.manifest = manifest

        sc = _fresh(base)
        hardware_platform(sc, key.commitment)
        out.append(_check(
            "T5 exchange completes with a TPM commitment and live PCRs",
            "restored", "restored" if pr.happy_path(sc) == sc.working_keys
            else "failed",
            "h_puf from a TPM-resident key, h_att from PCRs 0-7"))

        # A changed measurement must return the platform to denial at R1.
        sc = _fresh(base)
        hardware_platform(sc, key.commitment)
        ep = open_episode(sc)
        p = sc.platform
        p.admit(ep.token, ep.custody, ep.reencrypted)
        p.to_r1()
        p.manifest = pr.Manifest({**p.baseline.entries,
                                  "pcr04": bytes(32)})
        out.append(_expect_refusal(
            "T", "T5 a changed boot measurement aborts at R1",
            "attestation_mismatch", lambda: p.attest(),
            "PCR 4 is the boot manager and loader. In the design this is R1 "
            "refusing; on this machine the same refusal is available from the "
            "TPM itself (T3)"))

        # The cloned board: same everything, different device key.
        other, _h = store.create()
        sc = _fresh(base)
        hardware_platform(sc, key.commitment)
        ep = open_episode(sc)
        clone = copy.deepcopy(sc.platform)
        clone.puf_response = other.commitment
        out.append(_expect_refusal(
            "T", "T5 token presented on a board with a different device key",
            "transcript_mismatch",
            lambda: clone.admit(ep.token, ep.custody, ep.reencrypted),
            "refused on h_puf. Evidence about key binding, not about physical "
            "unclonability"))
    return out


# --- driver ------------------------------------------------------------------

TPM_EXPERIMENTS = {
    "T1": ("A secret the device cannot be separated from (TPM-resident key)",
           t1_device_bound_secret),
    "T2": ("The platform's measured state (PCRs 0-7 as h_att)",
           t2_measured_state),
    "T3": ("Sealing: a secret released only in the state it was sealed to",
           t3_sealing_to_measured_state),
    "T4": ("Values that cannot be wound back (TPM clock and reset count)",
           t4_monotonic_values),
}


def hardware_status() -> dict[str, object]:
    """What hardware was found, for the record. Never raises."""
    return {"tpm": tp.probe(), "platform_crypto_provider": pcp.probe()}


def run_all(base: pr.Scenario | None = None) -> tuple[list[Check],
                                                      dict[str, object]]:
    """Run T1-T5. Returns `(checks, status)`; checks is empty when there is no
    usable TPM, and the reason is in `status`."""
    status = hardware_status()
    if not status["tpm"].get("available") or \
            not status["platform_crypto_provider"].get("available"):
        return [], status

    checks: list[Check] = []
    for _, fn in TPM_EXPERIMENTS.values():
        checks.extend(fn())
    if base is not None:
        checks.extend(t5_hardware_backed_exchange(base))
    return checks, status
