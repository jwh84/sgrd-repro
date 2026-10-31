"""TPM-resident keys through the Windows Platform Crypto Provider, in ctypes.

Section `sec:design` binds each recovery exchange to the hull with
`h_puf`, a commitment to the platform instance's PUF challenge-response pair.
The commitment is not a key and the quorum never learns the response; what the
design needs from it is that *a token issued for one hull cannot recover a
cloned or replaced board*.

A TPM does not contain a PUF, and this module does not claim otherwise. What it
provides is the same architectural role discharged by different means: a key
pair whose private half is generated inside the TPM, is marked non-exportable,
and cannot be produced by any other machine. Experiment T1 establishes the
properties the design actually leans on -- the private half never appears
outside the chip, and the public half is a stable device commitment -- and says
plainly which property it does not establish, namely physical unclonability.

The provider is `Microsoft Platform Crypto Provider`, the CNG provider backed by
the TPM. Keys created through it are stored as TPM-wrapped blobs under the
user's profile and are unusable on any other TPM.
"""

from __future__ import annotations

import ctypes
import hashlib
import sys
import uuid
from ctypes import wintypes
from dataclasses import dataclass

PROVIDER = "Microsoft Platform Crypto Provider"
ALG_ECDSA_P256 = "ECDSA_P256"
BLOB_ECC_PUBLIC = "ECCPUBLICBLOB"
BLOB_ECC_PRIVATE = "ECCPRIVATEBLOB"

NCRYPT_EXPORT_POLICY_PROPERTY = "Export Policy"
NCRYPT_ALGORITHM_GROUP_PROPERTY = "Algorithm Group"

#: BCRYPT_ECDSA_PUBLIC_P256_MAGIC. The blob is magic(4) || cbKey(4) || X || Y.
ECDSA_P256_MAGIC = 0x31534345

ERROR_SUCCESS = 0


class PcpUnavailable(Exception):
    """No TPM-backed CNG provider. Raised so experiments can skip cleanly."""


class PcpError(Exception):
    def __init__(self, fn: str, status: int) -> None:
        super().__init__(f"{fn} -> 0x{status & 0xFFFFFFFF:08x}")
        self.fn = fn
        self.status = status


def _load() -> ctypes.WinDLL:
    if sys.platform != "win32":
        raise PcpUnavailable("CNG is a Windows interface")
    try:
        return ctypes.WinDLL("ncrypt.dll")
    except OSError as exc:  # pragma: no cover -- present on every Windows
        raise PcpUnavailable(f"ncrypt.dll not loadable: {exc}") from exc


def _check(fn: str, status: int) -> None:
    if status != ERROR_SUCCESS:
        raise PcpError(fn, status)


@dataclass(frozen=True)
class DeviceKey:
    """A TPM-resident signing key and the commitment derived from it."""

    name: str
    public_blob: bytes
    export_policy: int
    algorithm_group: str

    @property
    def commitment(self) -> bytes:
        """`h_dev`: the digest of the public key, standing in for `h_puf`.

        Like `h_puf` it is a commitment rather than a key, it is safe to publish,
        and it cannot be used to produce signatures. Unlike `h_puf` its
        uniqueness rests on the TPM's key generation and non-exportability
        rather than on manufacturing variation.
        """
        return hashlib.sha256(b"SGRD/v1/device|" + self.public_blob).digest()

    def public_numbers(self) -> tuple[int, int]:
        """`(x, y)` from the BCRYPT_ECCPUBLIC_BLOB, so that a signature can be
        verified by a library that never touched the TPM."""
        magic = int.from_bytes(self.public_blob[0:4], "little")
        if magic != ECDSA_P256_MAGIC:
            raise ValueError(f"unexpected key blob magic {magic:#x}")
        n = int.from_bytes(self.public_blob[4:8], "little")
        x = int.from_bytes(self.public_blob[8:8 + n], "big")
        y = int.from_bytes(self.public_blob[8 + n:8 + 2 * n], "big")
        return x, y


class PlatformKeyStore:
    """An open handle to the TPM-backed provider. Use as a context manager."""

    def __init__(self, provider: str = PROVIDER) -> None:
        self._n = _load()
        self._prov = ctypes.c_void_p()
        status = self._n.NCryptOpenStorageProvider(
            ctypes.byref(self._prov), ctypes.c_wchar_p(provider), 0)
        if status != ERROR_SUCCESS:
            raise PcpUnavailable(
                f"NCryptOpenStorageProvider({provider!r}) -> "
                f"0x{status & 0xFFFFFFFF:08x}")
        self._keys: list[tuple[str, ctypes.c_void_p]] = []

    def __enter__(self) -> "PlatformKeyStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Delete every key this store created.

        The experiments must leave nothing behind on the user's machine, so
        deletion happens here rather than being left to the caller's success
        path.
        """
        for _, h in self._keys:
            try:
                self._n.NCryptDeleteKey(h, 0)
            except OSError:  # pragma: no cover
                pass
        self._keys.clear()
        if self._prov:
            self._n.NCryptFreeObject(self._prov)
            self._prov = ctypes.c_void_p()

    # -- key lifecycle -------------------------------------------------------

    def create(self, algorithm: str = ALG_ECDSA_P256) -> tuple[DeviceKey,
                                                               ctypes.c_void_p]:
        """Generate a key inside the TPM and return it with its handle."""
        name = f"sgrd-t1-{uuid.uuid4().hex[:12]}"
        h = ctypes.c_void_p()
        _check("NCryptCreatePersistedKey", self._n.NCryptCreatePersistedKey(
            self._prov, ctypes.byref(h), ctypes.c_wchar_p(algorithm),
            ctypes.c_wchar_p(name), 0, 0))
        self._keys.append((name, h))
        _check("NCryptFinalizeKey", self._n.NCryptFinalizeKey(h, 0))
        key = DeviceKey(
            name=name,
            public_blob=self.export(h, BLOB_ECC_PUBLIC),
            export_policy=self._get_dword(h, NCRYPT_EXPORT_POLICY_PROPERTY),
            algorithm_group=self._get_wstr(h, NCRYPT_ALGORITHM_GROUP_PROPERTY),
        )
        return key, h

    def export(self, h: ctypes.c_void_p, blob_type: str) -> bytes:
        """Export a key blob. Raises :class:`PcpError` when the provider
        refuses -- which is the expected outcome for the private blob and the
        point of experiment T1."""
        need = wintypes.DWORD(0)
        _check("NCryptExportKey", self._n.NCryptExportKey(
            h, None, ctypes.c_wchar_p(blob_type), None, None, 0,
            ctypes.byref(need), 0))
        buf = ctypes.create_string_buffer(need.value)
        _check("NCryptExportKey", self._n.NCryptExportKey(
            h, None, ctypes.c_wchar_p(blob_type), None, buf, need.value,
            ctypes.byref(need), 0))
        return buf.raw[:need.value]

    def sign(self, h: ctypes.c_void_p, digest: bytes) -> bytes:
        """Sign a SHA-256 digest. Returns the raw `r || s` CNG produces."""
        need = wintypes.DWORD(0)
        _check("NCryptSignHash", self._n.NCryptSignHash(
            h, None, digest, len(digest), None, 0, ctypes.byref(need), 0))
        buf = ctypes.create_string_buffer(need.value)
        _check("NCryptSignHash", self._n.NCryptSignHash(
            h, None, digest, len(digest), buf, need.value,
            ctypes.byref(need), 0))
        return buf.raw[:need.value]

    # -- properties ----------------------------------------------------------

    def _get_dword(self, h: ctypes.c_void_p, prop: str) -> int:
        out = wintypes.DWORD(0)
        got = wintypes.DWORD(0)
        _check("NCryptGetProperty", self._n.NCryptGetProperty(
            h, ctypes.c_wchar_p(prop), ctypes.byref(out), 4,
            ctypes.byref(got), 0))
        return out.value

    def _get_wstr(self, h: ctypes.c_void_p, prop: str) -> str:
        got = wintypes.DWORD(0)
        _check("NCryptGetProperty", self._n.NCryptGetProperty(
            h, ctypes.c_wchar_p(prop), None, 0, ctypes.byref(got), 0))
        buf = ctypes.create_unicode_buffer(got.value // 2)
        _check("NCryptGetProperty", self._n.NCryptGetProperty(
            h, ctypes.c_wchar_p(prop), buf, got.value, ctypes.byref(got), 0))
        return buf.value


def probe() -> dict[str, object]:
    """Is a TPM-backed provider usable by this account? Never raises."""
    try:
        with PlatformKeyStore() as store:
            key, _h = store.create()
            return {
                "available": True,
                "provider": PROVIDER,
                "algorithm_group": key.algorithm_group,
                "export_policy": key.export_policy,
            }
    except (PcpUnavailable, PcpError, OSError) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
