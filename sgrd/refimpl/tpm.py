"""A minimal TPM 2.0 client over the Windows TPM Base Services, in ctypes.

Section `sec:impl` puts three of the design's elements above tier 0 because they
need hardware: a device-resident secret, a measurement of the platform's state,
and a counter that cannot be wound back. A TPM 2.0 provides analogues of all
three, and a discrete TPM is present on any machine that meets the Windows 11
installation requirement. That makes a middle tier possible -- not the tier that
settles the design's *own* hardware claims, but one that settles rather more
than software alone.

**What this module deliberately does not touch.** Nothing persistent. Keys are
created under `TPM_RH_NULL`, the hierarchy whose seed is regenerated at every
TPM reset and whose authorisation is always empty, so no owner or platform
credential is needed and nothing survives a reboot. The only PCR written is
PCR 23, which the TCG PC Client specification reserves for application use and
makes resettable from locality 0; it is read first, and the experiments refuse
to run if anything else has already extended it. No NV index is defined, no
hierarchy is cleared, no ownership is taken.

**Why no NV monotonic counter.** `TPM2_NV_DefineSpace` with `TPM_NT_COUNTER` is
what the design's counter `c` really wants. It requires owner or platform
authorisation and it consumes a scarce, persistent NV slot on the user's
machine. Both are reasons not to do it in an experiment. :meth:`Tpm.read_clock`
reads the TPM's clock and reset counter instead -- genuinely hardware-enforced
monotonic values that no software can wind back -- and Section `sec:impl` says
plainly that this substitutes for the counter rather than implementing it.

The marshalling is hand-rolled because the alternative is a dependency on
`tpm2-tools` or a TSS binding, neither of which is present by default on
Windows, and the command set needed here is small enough to write out. Every
structure below is laid out in the order TPM 2.0 Part 2 gives it.
"""

from __future__ import annotations

import ctypes
import hashlib
import sys
from dataclasses import dataclass
from typing import Iterable, Sequence

# --- constants (TPM 2.0 Part 2, Structures) ---------------------------------

ST_NO_SESSIONS = 0x8001
ST_SESSIONS = 0x8002

CC_PCR_EXTEND = 0x00000182
CC_PCR_READ = 0x0000017E
CC_PCR_RESET = 0x0000013D
CC_CREATE_PRIMARY = 0x00000131
CC_CREATE = 0x00000153
CC_LOAD = 0x00000157
CC_UNSEAL = 0x0000015E
CC_START_AUTH_SESSION = 0x00000176
CC_POLICY_PCR = 0x0000017F
CC_FLUSH_CONTEXT = 0x00000165
CC_READ_CLOCK = 0x00000181
CC_GET_CAPABILITY = 0x0000017A
CC_GET_RANDOM = 0x0000017B

RH_NULL = 0x40000007
RS_PW = 0x40000009

ALG_RSA = 0x0001
ALG_SHA256 = 0x000B
ALG_AES = 0x0006
ALG_KEYEDHASH = 0x0008
ALG_NULL = 0x0010
ALG_ECC = 0x0023
ALG_CFB = 0x0043

ECC_NIST_P256 = 0x0003
SE_POLICY = 0x01

#: Object attributes, TPM 2.0 Part 2 Table 31.
ATTR_FIXED_TPM = 1 << 1
ATTR_FIXED_PARENT = 1 << 4
ATTR_SENSITIVE_DATA_ORIGIN = 1 << 5
ATTR_USER_WITH_AUTH = 1 << 6
ATTR_NO_DA = 1 << 10
ATTR_RESTRICTED = 1 << 16
ATTR_DECRYPT = 1 << 17

#: The application PCR. TCG PC Client Platform Firmware Profile: PCR 23 is for
#: application use and is resettable from locality 0. Nothing in the boot chain
#: extends it, which is why it is the only PCR this module writes.
APPLICATION_PCR = 23

#: TPM_RC_POLICY_FAIL = RC_FMT1 + 0x01D. Format-one codes carry a handle,
#: parameter or session number in the upper bits, so the base error has to be
#: masked out rather than compared whole.
RC_FMT1 = 0x080
RC_POLICY_FAIL = RC_FMT1 + 0x01D


def is_policy_fail(rc: int) -> bool:
    """Did this response code mean "the policy was not satisfied"?

    The distinction matters to the experiment: a refusal on policy grounds is
    the result being looked for, whereas a refusal for any other reason is a
    bug in the test.
    """
    return bool(rc & RC_FMT1) and (rc & 0x03F) == 0x01D


#: TBS's own error space. Windows filters the command stream before it reaches
#: the TPM and returns these in the response-code field, so they have to be
#: told apart from anything the TPM itself said.
TBS_ERROR_MASK = 0xFFFF0000
TBS_ERROR_BASE = 0x80280000
TBS_E_BLOCKED = 0x80280400


class TpmUnavailable(Exception):
    """No usable TPM. Raised at open time so experiments can skip cleanly."""


class TpmError(Exception):
    """The TPM returned a non-zero response code."""

    def __init__(self, cc: int, rc: int) -> None:
        super().__init__(f"command 0x{cc:08x} -> TPM_RC 0x{rc:08x}")
        self.cc = cc
        self.rc = rc


class TpmBlocked(TpmError):
    """Windows TBS refused to pass the command to the TPM.

    Not a TPM verdict. The resource manager blocks a subset of commands from
    unprivileged callers -- PCR extension and reset among them -- so a
    `TpmBlocked` says something about this operating system's policy and nothing
    about the design under test.
    """

    def __init__(self, cc: int, rc: int) -> None:
        Exception.__init__(
            self, f"command 0x{cc:08x} blocked by Windows TBS (0x{rc:08x})")
        self.cc = cc
        self.rc = rc


# --- marshalling helpers -----------------------------------------------------

def u16(v: int) -> bytes:
    return v.to_bytes(2, "big")


def u32(v: int) -> bytes:
    return v.to_bytes(4, "big")


def tpm2b(data: bytes) -> bytes:
    """A TPM2B_ is a 16-bit length followed by that many bytes."""
    return u16(len(data)) + data


def read_tpm2b(buf: bytes, off: int) -> tuple[bytes, int]:
    n = int.from_bytes(buf[off:off + 2], "big")
    return buf[off + 2:off + 2 + n], off + 2 + n


#: A password session carrying an empty authorisation value: the auth area for
#: every command here that needs one, since nothing this module touches has a
#: non-empty auth.
_PW_SESSION = u32(RS_PW) + tpm2b(b"") + bytes([0]) + tpm2b(b"")
AUTH_EMPTY_PW = u32(len(_PW_SESSION)) + _PW_SESSION


def pcr_selection(indices: Iterable[int], hash_alg: int = ALG_SHA256) -> bytes:
    """A TPML_PCR_SELECTION over one bank.

    The three selection bytes are a little-endian bitmap over PCR index: bit `i`
    of byte `i // 8`. Getting this backwards is the classic first bug, and it
    fails silently by selecting the wrong PCRs rather than by erroring.
    """
    mask = bytearray(3)
    for i in indices:
        if not 0 <= i < 24:
            raise ValueError(f"PCR index out of range: {i}")
        mask[i // 8] |= 1 << (i % 8)
    return u32(1) + u16(hash_alg) + bytes([3]) + bytes(mask)


def policy_pcr_digest(pcr_values: dict[int, bytes],
                      hash_alg: int = ALG_SHA256) -> bytes:
    """The `authPolicy` an object must carry to be unsealable in that PCR state.

    TPM 2.0 Part 3, TPM2_PolicyPCR: starting from a zero policy digest,

        policy = H(policy || TPM_CC_PolicyPCR || pcrs || H(concatenated PCRs))

    Computing it here rather than asking the TPM is what lets the sealed object
    be created for a *stated* state. It is also the step that makes the
    correspondence with the paper exact: the object is bound to a measurement,
    not to a secret.
    """
    indices = sorted(pcr_values)
    concat = b"".join(pcr_values[i] for i in indices)
    digest_tpm = hashlib.sha256(concat).digest()
    return hashlib.sha256(
        bytes(32) + u32(CC_POLICY_PCR) + pcr_selection(indices, hash_alg)
        + digest_tpm).digest()


# --- the connection ----------------------------------------------------------

class _ContextParams2(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("flags", ctypes.c_uint32)]


@dataclass(frozen=True)
class ClockInfo:
    """TPMS_TIME_INFO. `clock` and `reset_count` are the hardware-enforced
    monotonic values; `safe` reports whether the TPM can guarantee that `clock`
    has never gone backwards."""

    time_ms: int
    clock_ms: int
    reset_count: int
    restart_count: int
    safe: bool


@dataclass(frozen=True)
class Sealed:
    """What `TPM2_Create` returns for a sealed data object: two opaque blobs
    that only this TPM can load, and only in the sealed PCR state can open."""

    private: bytes
    public: bytes


class Tpm:
    """A TBS connection. Use as a context manager; transient handles are
    flushed on exit even if a command raised."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise TpmUnavailable("TBS is a Windows interface")
        try:
            self._tbs = ctypes.WinDLL("tbs.dll")
        except OSError as exc:
            raise TpmUnavailable(f"tbs.dll not loadable: {exc}") from exc
        self._ctx = ctypes.c_void_p()
        params = _ContextParams2(2, 4)  # TBS_CONTEXT_VERSION_TWO, includeTpm20
        rc = self._tbs.Tbsi_Context_Create(ctypes.byref(params),
                                           ctypes.byref(self._ctx))
        if rc != 0:
            raise TpmUnavailable(f"Tbsi_Context_Create 0x{rc & 0xFFFFFFFF:08x}")
        self._open_handles: list[int] = []

    def __enter__(self) -> "Tpm":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for h in reversed(self._open_handles):
            try:
                self.flush(h, _track=False)
            except (TpmError, OSError):
                pass  # closing the context frees them anyway
        self._open_handles.clear()
        if self._ctx:
            self._tbs.Tbsip_Context_Close(self._ctx)
            self._ctx = ctypes.c_void_p()

    # -- transport -----------------------------------------------------------

    def submit(self, cc: int, body: bytes = b"", *, tag: int = ST_NO_SESSIONS,
               allow_rc: Sequence[int] = ()) -> tuple[int, bytes]:
        """Send one command. Returns `(rc, response body after the header)`.

        `allow_rc` names response codes the caller expects, so that a
        deliberately failing unseal is a result rather than an exception.
        """
        cmd = u16(tag) + u32(10 + len(body)) + u32(cc) + body
        buf = ctypes.create_string_buffer(8192)
        size = ctypes.c_uint32(8192)
        rc = self._tbs.Tbsip_Submit_Command(
            self._ctx, 0, 200, cmd, len(cmd), buf, ctypes.byref(size))
        if rc != 0:
            raise OSError(f"Tbsip_Submit_Command 0x{rc & 0xFFFFFFFF:08x}")
        resp = buf.raw[:size.value]
        tpm_rc = int.from_bytes(resp[6:10], "big")
        if tpm_rc != 0 and tpm_rc not in allow_rc:
            if tpm_rc & TBS_ERROR_MASK == TBS_ERROR_BASE:
                raise TpmBlocked(cc, tpm_rc)
            raise TpmError(cc, tpm_rc)
        return tpm_rc, resp[10:]

    # -- capability and identity --------------------------------------------

    def properties(self) -> dict[int, int]:
        """TPM_CAP_TPM_PROPERTIES from the fixed group, as `{property: value}`."""
        _, body = self.submit(CC_GET_CAPABILITY,
                              u32(6) + u32(0x100) + u32(64))
        n = int.from_bytes(body[5:9], "big")
        off = 9
        out: dict[int, int] = {}
        for _ in range(n):
            out[int.from_bytes(body[off:off + 4], "big")] = \
                int.from_bytes(body[off + 4:off + 8], "big")
            off += 8
        return out

    def identity(self) -> dict[str, str]:
        """Manufacturer, part number and firmware version, for the record.

        Section `sec:impl` requires the hardware to be named in any measurement
        it reports, so this is read and printed rather than left to the reader.
        """
        p = self.properties()

        def ascii4(v: int) -> str:
            return v.to_bytes(4, "big").decode("ascii", "replace")

        vendor = "".join(ascii4(p.get(k, 0))
                         for k in (0x106, 0x107, 0x108, 0x109))
        return {
            "manufacturer": ascii4(p.get(0x105, 0)).strip("\x00 "),
            "vendor_string": vendor.strip("\x00 "),
            "firmware_version": f"{p.get(0x10B, 0):#010x}.{p.get(0x10C, 0):#010x}",
            "spec_revision": str(p.get(0x102, 0)),
        }

    def get_random(self, n: int) -> bytes:
        _, body = self.submit(CC_GET_RANDOM, u16(n))
        data, _ = read_tpm2b(body, 0)
        return data

    # -- the monotonic values ------------------------------------------------

    def read_clock(self) -> ClockInfo:
        """TPM2_ReadClock. No authorisation, no state change.

        `clock` counts milliseconds the TPM has been powered since it was last
        cleared and is required by the specification never to decrease -- the
        TPM persists it across power cycles itself. `reset_count` increments on
        every TPM reset. Neither can be wound back by software, which is the
        property the design's counter `c` needs.
        """
        _, body = self.submit(CC_READ_CLOCK)
        return ClockInfo(
            time_ms=int.from_bytes(body[0:8], "big"),
            clock_ms=int.from_bytes(body[8:16], "big"),
            reset_count=int.from_bytes(body[16:20], "big"),
            restart_count=int.from_bytes(body[20:24], "big"),
            safe=bool(body[24]),
        )

    # -- PCRs ----------------------------------------------------------------

    def pcr_read(self, indices: Sequence[int]) -> dict[int, bytes]:
        """Read PCRs one at a time.

        The TPM truncates a multi-PCR read to whatever fits in one response and
        silently narrows the selection it echoes back; reading singly costs a
        command apiece and removes the chance of attributing one PCR's value to
        another.
        """
        out: dict[int, bytes] = {}
        for i in indices:
            _, body = self.submit(CC_PCR_READ, pcr_selection([i]))
            # pcrUpdateCounter(4) TPML_PCR_SELECTION(4+2+1+3) TPML_DIGEST
            count = int.from_bytes(body[14:18], "big")
            if count != 1:
                raise TpmError(CC_PCR_READ, 0)
            digest, _ = read_tpm2b(body, 18)
            out[i] = digest
        return out

    def pcr_extend(self, index: int, digest: bytes) -> None:
        """Extend a PCR: `PCR := H(PCR || digest)`.

        **Windows blocks this from user mode.** TBS filters PCR extension and
        reset out of the unprivileged command stream and returns
        :data:`TBS_E_BLOCKED`, so this raises :class:`TpmBlocked` rather than
        doing anything. It is kept, and called once by experiment T3, because
        the block is itself worth recording: it is why T3 demonstrates the
        measurement binding by sealing to *differing* states rather than by
        mutating one, and that turns out to be the better experiment, since it
        uses the real boot-chain PCRs instead of the scratch one.
        """
        if len(digest) != 32:
            raise ValueError("SHA-256 bank takes a 32-byte digest")
        self.submit(CC_PCR_EXTEND, tag=ST_SESSIONS,
                    body=u32(index) + AUTH_EMPTY_PW
                    + u32(1) + u16(ALG_SHA256) + digest)

    def pcr_reset(self, index: int) -> None:
        """Reset a resettable PCR. Blocked from user mode; see
        :meth:`pcr_extend`."""
        self.submit(CC_PCR_RESET, tag=ST_SESSIONS,
                    body=u32(index) + AUTH_EMPTY_PW)

    # -- objects -------------------------------------------------------------

    def create_primary_null(self) -> int:
        """An ECC P-256 storage parent under `TPM_RH_NULL`.

        ECC rather than RSA because a discrete TPM generates an RSA-2048 primary
        in seconds to minutes and a P-256 one in milliseconds, and nothing here
        depends on which. The null hierarchy because its authorisation is always
        empty and its seed is discarded at every TPM reset -- so this leaves
        nothing behind, and a deployment would instead use a persistent parent
        under the owner hierarchy.
        """
        attrs = (ATTR_FIXED_TPM | ATTR_FIXED_PARENT | ATTR_SENSITIVE_DATA_ORIGIN
                 | ATTR_USER_WITH_AUTH | ATTR_RESTRICTED | ATTR_DECRYPT)
        public = (
            u16(ALG_ECC) + u16(ALG_SHA256) + u32(attrs) + tpm2b(b"")
            + u16(ALG_AES) + u16(128) + u16(ALG_CFB)   # symmetric, for wrapping
            + u16(ALG_NULL)                            # scheme
            + u16(ECC_NIST_P256)
            + u16(ALG_NULL)                            # kdf
            + tpm2b(b"") + tpm2b(b"")                  # unique point
        )
        body = (u32(RH_NULL) + AUTH_EMPTY_PW
                + tpm2b(tpm2b(b"") + tpm2b(b""))       # inSensitive
                + tpm2b(public) + tpm2b(b"") + u32(0))
        _, resp = self.submit(CC_CREATE_PRIMARY, body, tag=ST_SESSIONS)
        handle = int.from_bytes(resp[0:4], "big")
        self._open_handles.append(handle)
        return handle

    def seal(self, parent: int, secret: bytes, policy: bytes) -> Sealed:
        """Seal `secret` so that it opens only under a session satisfying
        `policy`.

        `userWithAuth` is left CLEAR so that USER-role authorisation -- which is
        what `TPM2_Unseal` needs -- *requires* a policy session. Leaving it set
        would let an empty password open the object, and the whole experiment
        would pass for the wrong reason.

        `noDA` is set. The experiment deliberately provokes a policy failure,
        and without `noDA` that failure would count towards the TPM's dictionary
        attack lockout on the user's own machine.
        """
        attrs = ATTR_FIXED_TPM | ATTR_FIXED_PARENT | ATTR_NO_DA
        public = (u16(ALG_KEYEDHASH) + u16(ALG_SHA256) + u32(attrs)
                  + tpm2b(policy) + u16(ALG_NULL) + tpm2b(b""))
        body = (u32(parent) + AUTH_EMPTY_PW
                + tpm2b(tpm2b(b"") + tpm2b(secret))
                + tpm2b(public) + tpm2b(b"") + u32(0))
        _, resp = self.submit(CC_CREATE, body, tag=ST_SESSIONS)
        private, off = read_tpm2b(resp, 4)   # after parameterSize
        public_out, _ = read_tpm2b(resp, off)
        return Sealed(private=private, public=public_out)

    def load(self, parent: int, sealed: Sealed) -> int:
        body = (u32(parent) + AUTH_EMPTY_PW
                + tpm2b(sealed.private) + tpm2b(sealed.public))
        _, resp = self.submit(CC_LOAD, body, tag=ST_SESSIONS)
        handle = int.from_bytes(resp[0:4], "big")
        self._open_handles.append(handle)
        return handle

    def flush(self, handle: int, *, _track: bool = True) -> None:
        self.submit(CC_FLUSH_CONTEXT, u32(handle))
        if _track and handle in self._open_handles:
            self._open_handles.remove(handle)

    # -- policy sessions -----------------------------------------------------

    def start_policy_session(self) -> int:
        body = (u32(RH_NULL) + u32(RH_NULL)
                + tpm2b(self.get_random(32))   # nonceCaller
                + tpm2b(b"")                   # encryptedSalt
                + bytes([SE_POLICY])
                + u16(ALG_NULL)                # symmetric
                + u16(ALG_SHA256))
        _, resp = self.submit(CC_START_AUTH_SESSION, body)
        handle = int.from_bytes(resp[0:4], "big")
        self._open_handles.append(handle)
        return handle

    def policy_pcr(self, session: int, indices: Sequence[int]) -> None:
        """Bind the session to the PCRs' *current* values.

        The `pcrDigest` argument is left empty, which tells the TPM to read the
        PCRs itself and fold in what it finds. That is the whole mechanism: the
        session's policy digest becomes a function of the machine's measured
        state, and it will match the sealed object's `authPolicy` only if that
        state is the one the object was sealed to.
        """
        self.submit(CC_POLICY_PCR,
                    u32(session) + tpm2b(b"") + pcr_selection(indices))

    def unseal(self, item: int, session: int) -> bytes | None:
        """Attempt `TPM2_Unseal`. Returns the secret, or `None` on policy
        failure -- the outcome the experiment is looking for."""
        auth = (u32(session) + tpm2b(b"") + bytes([0x01]) + tpm2b(b""))
        body = u32(item) + u32(len(auth)) + auth
        try:
            rc, resp = self.submit(CC_UNSEAL, body, tag=ST_SESSIONS)
        except TpmError as exc:
            if is_policy_fail(exc.rc):
                return None
            raise
        data, _ = read_tpm2b(resp, 4)
        return data


def probe() -> dict[str, object]:
    """Is there a usable TPM? Never raises, so callers can skip cleanly."""
    try:
        with Tpm() as t:
            info = t.identity()
            clock = t.read_clock()
            pcr23 = t.pcr_read([APPLICATION_PCR])[APPLICATION_PCR]
            return {
                "available": True,
                **info,
                "reset_count": clock.reset_count,
                "clock_safe": clock.safe,
                "application_pcr_unextended": pcr23 == bytes(32),
            }
    except (TpmUnavailable, TpmError, OSError) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
