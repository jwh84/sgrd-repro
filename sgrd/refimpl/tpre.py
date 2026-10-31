"""Threshold proxy re-encryption: one instantiation of the three requirements
Section `sec:design` places on the recovery-capsule path.

The paper declines to propose a suite and instead states what a candidate must
do:

1. `t-1` holders must not learn `K_e`;
2. the combiner must learn nothing from what it assembles;
3. a malformed share must be *detectable*, not merely fatal.

Requirement 2 is the one that rules out the obvious construction. Plain
threshold decryption has the holders emit partial decryptions and the combiner
assemble them into plaintext `K_e` -- at which point the combiner is an
off-platform party holding the very key the design puts beyond reach. This
module therefore has each holder blind its contribution *under the platform's
ephemeral key* before releasing it, so the combiner assembles a ciphertext it
cannot open.

Concretely, with `K_r = x`, `pk_r = g^x`, Shamir shares `s_i` of `x`, capsule
element `c1 = g^r`, and platform ephemeral key `pk_eph = g^y`:

    holder i:   D_i = c1^(s_i) * pk_eph^(k_i)     E_i = g^(k_i)      k_i fresh
    combiner:   Zmask = prod D_i^(lambda_i)       Emask = prod E_i^(lambda_i)
                      = Z * pk_eph^K                    = g^K
    platform:   Z = Zmask / Emask^y

where `Z = g^(rx) = pk_r^r` is the KEM shared secret and `K = sum lambda_i k_i`
is known to no single party. The combiner holds `Z * pk_eph^K` and `g^K`;
recovering `Z` from those is exactly a decisional Diffie-Hellman problem, and
the ablation in experiment S2 runs the same code with the blinding removed to
show that this is what does the work rather than an incidental feature.

Requirement 3 is discharged by a Fiat-Shamir'd sigma protocol: each holder
proves knowledge of `(s_i, k_i)` consistent with `D_i`, `E_i`, and the
verification key `V_i = g^(s_i)` published at provisioning. A wrong share fails
its own proof, so the combiner can name the holder rather than merely observing
that the assembly did not open.

Nothing here is novel; the pieces are Shamir, ElGamal, Blaze-Bleumer-Strauss,
and Chaum-Pedersen. Novelty is not what a reference implementation is for.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from . import group as gp

DOMAIN_KEM = b"SGRD/v1/capsule-kem"
DOMAIN_PROOF = b"SGRD/v1/reshare-proof"


class ShareRejected(Exception):
    """A re-encryption share failed its own proof. Carries the holder index, so
    the combiner can report *who* rather than only *that* -- requirement 3."""

    def __init__(self, index: int, reason: str) -> None:
        super().__init__(f"holder {index}: {reason}")
        self.index = index
        self.reason = reason


# --- provisioning ------------------------------------------------------------

@dataclass(frozen=True)
class HolderShare:
    """What one holder `Q_i` keeps. Never leaves that holder."""

    index: int
    value: int


@dataclass(frozen=True)
class RecoveryAuthority:
    """The public half of the distributed recovery authority.

    `K_r` itself is *not* a field of anything in this module. It exists for one
    instant inside :func:`provision` and is never returned, which is the
    executable form of the paper's claim that it is never reconstituted as a
    whole.
    """

    pk_r: int
    t: int
    n: int
    verification_keys: dict[int, int]  # index -> g^(s_i)


def provision(t: int, n: int) -> tuple[RecoveryAuthority, list[HolderShare]]:
    """Generate the authority and hand each holder its share.

    The secret `x` is local to this function and goes out of scope on return.
    That is a modelling convenience, not a security property -- a real
    provisioning ceremony would run distributed key generation so that `x` never
    exists anywhere. Section `sec:disc` records this as an obligation the paper
    does not discharge; the reference implementation inherits it.
    """
    x = gp.rand_scalar()
    pk_r = gp.exp(gp.G, x)
    shares = [HolderShare(i, v) for i, v in gp.share(x, t, n)]
    vks = {s.index: gp.exp(gp.G, s.value) for s in shares}
    return RecoveryAuthority(pk_r=pk_r, t=t, n=n, verification_keys=vks), shares


# --- the capsule -------------------------------------------------------------

@dataclass(frozen=True)
class Capsule:
    """`RC = Enc_{pk_r}(K_e || epoch)` of Section `sec:design`, as a KEM/DEM
    pair. The platform can build one and cannot open it: sealing needs `pk_r`,
    opening needs `x`, and the platform has only the former."""

    c1: int  # g^r
    nonce: bytes
    ct: bytes  # AES-256-GCM over epoch || K_e
    vessel_id: bytes
    epoch: int

    def digest(self) -> bytes:
        """The capsule's identity for transcript binding. `T` commits to this,
        which is what stops a token from authorising a *different* capsule."""
        h = hashes.Hash(hashes.SHA256())
        for part in (gp.i2osp(self.c1), self.nonce, self.ct, self.vessel_id,
                     self.epoch.to_bytes(8, "big")):
            h.update(len(part).to_bytes(4, "big"))
            h.update(part)
        return h.finalize()


def _kem_key(z: int, label: bytes) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=DOMAIN_KEM + b"|" + label).derive(gp.i2osp(z))


def seal(authority: RecoveryAuthority, kek: bytes, *, vessel_id: bytes,
         epoch: int) -> Capsule:
    """Commit a capsule for the current epoch.

    Called at provisioning and again at every R3 as soon as a fresh `K_e` is
    drawn -- never at denial time. Section `sec:design` is explicit that the
    reverse order is incoherent: a platform that mints the capsule while denial
    fires must still hold `K_e` in the clear at that moment, which contradicts
    the state denial is trying to reach.
    """
    r = gp.rand_scalar()
    c1 = gp.exp(gp.G, r)
    z = gp.exp(authority.pk_r, r)
    nonce = os.urandom(12)
    aad = vessel_id + b"|" + epoch.to_bytes(8, "big")
    pt = epoch.to_bytes(8, "big") + kek
    ct = AESGCM(_kem_key(z, aad)).encrypt(nonce, pt, aad)
    return Capsule(c1=c1, nonce=nonce, ct=ct, vessel_id=vessel_id, epoch=epoch)


# --- re-encryption shares ----------------------------------------------------

@dataclass(frozen=True)
class ReShare:
    """One holder's contribution, with the proof that makes it checkable."""

    index: int
    d: int  # c1^(s_i) * pk_eph^(k_i)
    e: int  # g^(k_i)
    # Fiat-Shamir transcript: commitments then responses.
    a: int
    b: int
    c: int
    z_s: int
    z_k: int


def reshare(share_: HolderShare, capsule: Capsule, pk_eph: int, *,
            blind: bool = True) -> ReShare:
    """Produce holder `i`'s re-encryption share, targeted at `pk_eph`.

    `blind=False` is the ablation of experiment S2. It reduces the construction
    to ordinary threshold decryption, which satisfies requirements 1 and 3 and
    fails requirement 2 -- the combiner recovers `K_e`. It exists so that the
    claim "the blinding is what keeps the combiner ignorant" can be exhibited
    rather than asserted, in the same way Section `sec:results` ablates the
    controller flags rather than reporting a bare 12-of-12.
    """
    k = gp.rand_scalar() if blind else 0
    d = gp.mul(gp.exp(capsule.c1, share_.value), gp.exp(pk_eph, k))
    e = gp.exp(gp.G, k)

    rho_s, rho_k = gp.rand_scalar(), gp.rand_scalar()
    a = gp.mul(gp.exp(capsule.c1, rho_s), gp.exp(pk_eph, rho_k))
    b = gp.exp(gp.G, rho_k)
    c = gp.exp(gp.G, rho_s)
    chal = gp.hash_to_scalar(
        DOMAIN_PROOF, share_.index, capsule.c1, pk_eph,
        gp.exp(gp.G, share_.value), d, e, a, b, c, capsule.digest())
    return ReShare(
        index=share_.index, d=d, e=e, a=a, b=b, c=c,
        z_s=(rho_s + chal * share_.value) % gp.Q,
        z_k=(rho_k + chal * k) % gp.Q,
    )


def verify_reshare(rs: ReShare, authority: RecoveryAuthority, capsule: Capsule,
                   pk_eph: int) -> None:
    """Check one share. Raises :class:`ShareRejected` naming the holder.

    Requirement 3 of Section `sec:design`. Without this the combiner sees only
    that the assembled capsule does not open, and cannot say which of the `t`
    holders is at fault -- which in an operational setting means the recovery
    stalls with no one accountable.
    """
    vk = authority.verification_keys.get(rs.index)
    if vk is None:
        raise ShareRejected(rs.index, "not a registered holder")
    for name, val in (("d", rs.d), ("e", rs.e), ("a", rs.a), ("b", rs.b),
                      ("c", rs.c)):
        if not gp.is_member(val):
            raise ShareRejected(rs.index, f"{name} is outside the group")

    chal = gp.hash_to_scalar(
        DOMAIN_PROOF, rs.index, capsule.c1, pk_eph, vk, rs.d, rs.e,
        rs.a, rs.b, rs.c, capsule.digest())
    checks = (
        # knows s_i, k_i with D_i = c1^s_i * pk_eph^k_i
        gp.mul(gp.exp(capsule.c1, rs.z_s), gp.exp(pk_eph, rs.z_k))
        == gp.mul(rs.a, gp.exp(rs.d, chal)),
        # the same k_i as in E_i
        gp.exp(gp.G, rs.z_k) == gp.mul(rs.b, gp.exp(rs.e, chal)),
        # the same s_i as the registered verification key
        gp.exp(gp.G, rs.z_s) == gp.mul(rs.c, gp.exp(vk, chal)),
    )
    if not all(checks):
        which = [n for n, ok in zip(("D_i", "E_i", "V_i"), checks) if not ok]
        raise ShareRejected(rs.index, f"proof failed on {', '.join(which)}")


# --- combination -------------------------------------------------------------

@dataclass(frozen=True)
class ReEncrypted:
    """What the combiner assembles and hands to the platform.

    This object *is* the combiner's whole view of the secret material. Recovering
    `K_e` from it is a decisional Diffie-Hellman problem; experiment S2 tests
    the structural consequence, which is that opening it requires `sk_eph`.
    """

    z_masked: int  # Z * pk_eph^K
    e_masked: int  # g^K
    contributors: tuple[int, ...]


def combine(shares: Sequence[ReShare], authority: RecoveryAuthority,
            capsule: Capsule, pk_eph: int) -> ReEncrypted:
    """Assemble `t` verified shares. Every share is checked before use."""
    if len({s.index for s in shares}) != len(shares):
        raise ValueError("duplicate holder indices")
    for rs in shares:
        verify_reshare(rs, authority, capsule, pk_eph)

    lam = gp.lagrange_coefficients([s.index for s in shares])
    z_masked, e_masked = 1, 1
    for rs in shares:
        z_masked = gp.mul(z_masked, gp.exp(rs.d, lam[rs.index]))
        e_masked = gp.mul(e_masked, gp.exp(rs.e, lam[rs.index]))
    return ReEncrypted(z_masked=z_masked, e_masked=e_masked,
                       contributors=tuple(sorted(s.index for s in shares)))


def unwrap(re: ReEncrypted, sk_eph: int, capsule: Capsule) -> bytes:
    """Recover `K_e` inside the module boundary.

    Raises `cryptography.exceptions.InvalidTag` when the assembly was built from
    too few shares, from the wrong capsule, or for a different ephemeral key.
    That the failure surfaces as an authentication-tag rejection rather than as
    garbage plaintext is the point: below the threshold there is no partial
    answer, only a refusal.
    """
    z = gp.div(re.z_masked, gp.exp(re.e_masked, sk_eph))
    aad = capsule.vessel_id + b"|" + capsule.epoch.to_bytes(8, "big")
    pt = AESGCM(_kem_key(z, aad)).decrypt(capsule.nonce, capsule.ct, aad)
    epoch = int.from_bytes(pt[:8], "big")
    if epoch != capsule.epoch:
        raise ValueError("capsule epoch does not match its own header")
    return pt[8:]
