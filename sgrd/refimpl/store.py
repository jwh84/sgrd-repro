"""Non-volatile storage for the recovery capsule, and the commit protocol
Section `sec:design` requires to be atomic.

The design's sentence is short and its consequences are not:

    The module zeroises the previous epoch's `K_e` and capsule only after the
    new capsule is completely written to non-volatile storage and its digest
    confirmed; if power is lost in between, at restart it compares the monotonic
    counter against the capsule's epoch and rolls back the incomplete update.
    So at whatever instant power is lost, the platform always holds exactly one
    coherent (key, capsule) pair.

That is a claim about an algorithm *and* a claim about a storage medium, and the
two need separating because only one of them can be settled here.

**The algorithm.** Two slots, alternating, each a self-describing record under a
digest; a counter that names which slot is committed. Advancing the counter is
the commit point, and it is the only irreversible step. Everything before it is
undoable by discarding a slot; everything after it is idempotent. Experiments
A1-A4 exercise every write and every sync boundary in that sequence.

**The medium.** Whether a write that returned actually reached the platters,
whether a sector can tear, whether the flash translation layer reorders --
none of that is visible from user space, and none of it is tested here. What
this module does instead is make the algorithm's requirements on the medium
*explicit*, as :data:`MEDIUM_REQUIREMENTS`, so that they can be checked against
a datasheet rather than assumed.

**On the counter.** The design's `c` is a hardware monotonic counter -- a TPM NV
index of type `TPM_NT_COUNTER`, or the equivalent in a secure element.
Experiment T4 showed the hardware offers such a value. Here it is modelled by a
pair of digest-protected records, alternating the same way the slots do, so that
a torn counter write is detectable rather than fatal. A deployment would replace
this with the hardware counter and inherit its guarantee instead.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

MAGIC_SLOT = b"SGRD-CAP"
MAGIC_CTR = b"SGRD-CTR"

#: What the algorithm below assumes of the storage layer. Section `sec:impl`
#: states these as requirements the mechanism places on the platform, in the
#: same way Section `sec:design` states the memory-encryption boundary -- not as
#: assumptions about a platform, and not as things this work has verified.
MEDIUM_REQUIREMENTS = (
    "a write reported complete is durable, or is reported not complete",
    "writes to distinct files are not reordered across an intervening sync",
    "a torn write may corrupt a record but must not corrupt a different record",
    "the counter's two records do not share a failure domain with each other",
)


class PowerLoss(Exception):
    """The simulated device lost power. Not an error: the outcome under test."""


# --- records -----------------------------------------------------------------

def _digest(body: bytes) -> bytes:
    return hashlib.sha256(body).digest()


def encode_slot(epoch: int, wrapped: bytes, capsule: bytes) -> bytes:
    """A slot record: magic, epoch, the two blobs, and a digest over all of it.

    Self-describing and self-checking, so that a partially written record is
    rejected by its own digest rather than by anything the reader has to
    remember about how big it should have been.
    """
    body = (MAGIC_SLOT + epoch.to_bytes(8, "big")
            + len(wrapped).to_bytes(4, "big") + wrapped
            + len(capsule).to_bytes(4, "big") + capsule)
    return body + _digest(body)


def decode_slot(raw: bytes) -> tuple[int, bytes, bytes] | None:
    """Parse a slot, or `None` if it is torn, empty, or not a slot at all."""
    try:
        if len(raw) < len(MAGIC_SLOT) + 8 + 32 or raw[:8] != MAGIC_SLOT:
            return None
        body, tag = raw[:-32], raw[-32:]
        if _digest(body) != tag:
            return None
        off = len(MAGIC_SLOT)
        epoch = int.from_bytes(body[off:off + 8], "big")
        off += 8
        n = int.from_bytes(body[off:off + 4], "big")
        off += 4
        wrapped = body[off:off + n]
        off += n
        m = int.from_bytes(body[off:off + 4], "big")
        off += 4
        capsule = body[off:off + m]
        if off + m != len(body):
            return None
        return epoch, wrapped, capsule
    except (IndexError, ValueError):
        return None


def encode_counter(value: int, slot: int) -> bytes:
    body = MAGIC_CTR + value.to_bytes(8, "big") + bytes([slot])
    return body + _digest(body)


def decode_counter(raw: bytes) -> tuple[int, int] | None:
    if len(raw) != len(MAGIC_CTR) + 9 + 32 or raw[:8] != MAGIC_CTR:
        return None
    body, tag = raw[:-32], raw[-32:]
    if _digest(body) != tag:
        return None
    return int.from_bytes(body[8:16], "big"), body[16]


# --- devices -----------------------------------------------------------------

@dataclass
class Fault:
    """When to cut power, and how.

    `at` counts device operations from 1. `mode` is one of:

    ``cut``     the operation does not happen at all
    ``tear``    a write happens, but only its first `prefix` bytes

    What survives the cut is not part of the fault: it is
    :attr:`SimDevice.survive`, enumerated separately, because the question
    "which pending writes reached the medium" has an answer per subset rather
    than per fault. Neither can be produced by killing a process -- the kernel
    outlives the writer and flushes its page cache regardless of what happened
    to it -- and that is why the simulated device exists alongside A4's real one.
    """

    at: int
    mode: str = "cut"
    prefix: int = 0


class SimDevice:
    """A block device that can be made to lose power at a named instant.

    Keeps two copies of every file: what is durable, and what has been written
    but not yet synced. A power cut discards the second, which is the behaviour
    a real device is permitted -- and the behaviour a killed process cannot
    exhibit, because the kernel outlives it.
    """

    def __init__(self, fault: Fault | None = None) -> None:
        self.durable: dict[str, bytes] = {}
        self.pending: dict[str, bytes] = {}
        self.fault = fault
        self.ops = 0
        self.trace: list[str] = []
        #: Which pending writes reach the medium anyway when power is cut.
        #: The caller enumerates the subsets; see :meth:`power_cut`.
        self.survive: frozenset[str] = frozenset()

    # -- fault plumbing ------------------------------------------------------

    def _tick(self, what: str) -> str | None:
        self.ops += 1
        self.trace.append(f"{self.ops}:{what}")
        if self.fault is None or self.ops != self.fault.at:
            return None
        return self.fault.mode

    def power_cut(self, survive: frozenset[str] = frozenset()) -> None:
        """Lose power. Pending writes named in `survive` reached the medium
        anyway; the rest did not.

        Modelling it as an arbitrary subset rather than "everything pending is
        lost" is what makes reordering visible. A medium is free to write out
        what it holds in any order, so between two unsynced writes either, both
        or neither may survive -- and a protocol that needs the first to land
        before the second has to say so with a sync. Experiment A3 enumerates
        the subsets; with every sync in place at most one write is ever pending
        and the enumeration is trivial, which is the point.
        """
        for name in list(self.pending):
            if name in survive:
                self.durable[name] = self.pending[name]
            del self.pending[name]

    # -- device interface ----------------------------------------------------

    #: Set by the caller before a run: which pending writes survive the cut.
    survive: frozenset[str] = frozenset()

    def write(self, name: str, data: bytes) -> None:
        mode = self._tick(f"write {name}")
        if mode == "cut":
            self.power_cut(self.survive)
            raise PowerLoss(f"cut before write {name}")
        if mode == "tear":
            # The torn prefix reached the medium; nothing else pending did,
            # unless the caller said it survives.
            self.durable[name] = data[:self.fault.prefix]
            self.pending.pop(name, None)
            self.power_cut(self.survive)
            raise PowerLoss(f"torn write {name} at {self.fault.prefix}")
        self.pending[name] = data

    def sync(self, name: str) -> None:
        mode = self._tick(f"sync {name}")
        if mode == "cut":
            self.power_cut(self.survive)
            raise PowerLoss(f"cut before sync {name}")
        if name in self.pending:
            self.durable[name] = self.pending.pop(name)

    def read(self, name: str) -> bytes:
        # A reader sees the durable image plus anything this process has written
        # since; after a power cut the pending map is empty, so it sees only
        # what survived.
        return self.pending.get(name, self.durable.get(name, b""))

    def snapshot(self) -> "SimDevice":
        """A fresh device holding only what survived, for the restart."""
        d = SimDevice()
        d.durable = dict(self.durable)
        return d


class RealDevice:
    """Actual files, for the process-kill experiment.

    Writes are `os.write` to a freshly opened descriptor and syncs are
    `os.fsync`, so nothing sits in a Python buffer waiting for an interpreter
    that is about to stop existing.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.ops = 0
        self.fault: Fault | None = None

    def _path(self, name: str) -> str:
        return os.path.join(self.root, name)

    def _tick(self, _what: str) -> str | None:
        self.ops += 1
        if self.fault is None or self.ops != self.fault.at:
            return None
        return self.fault.mode

    #: Windows opens descriptors in text mode by default, which would translate
    #: a 0x0A inside a digest into 0x0D 0x0A and corrupt every record that
    #: happened to contain one. Absent on POSIX, where it is a no-op.
    _BINARY = getattr(os, "O_BINARY", 0)

    def write(self, name: str, data: bytes) -> None:
        mode = self._tick(name)
        if mode == "cut":
            os._exit(9)
        if mode == "tear":
            data = data[:self.fault.prefix]
        fd = os.open(self._path(name),
                     os.O_WRONLY | os.O_CREAT | os.O_TRUNC | self._BINARY)
        try:
            os.write(fd, data)
            if mode == "tear":
                os.fsync(fd)
        finally:
            os.close(fd)
        if mode == "tear":
            os._exit(9)

    def sync(self, name: str) -> None:
        mode = self._tick(name)
        if mode == "cut":
            os._exit(9)
        # Read-write, not read-only: Windows refuses to flush the buffers of a
        # handle that was not opened for writing.
        fd = os.open(self._path(name), os.O_RDWR | self._BINARY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def read(self, name: str) -> bytes:
        try:
            with open(self._path(name), "rb") as fh:
                return fh.read()
        except FileNotFoundError:
            return b""


Device = SimDevice | RealDevice


# --- the store ---------------------------------------------------------------

SLOTS = ("slot0", "slot1")
COUNTERS = ("ctr0", "ctr1")


@dataclass(frozen=True)
class State:
    """What a restart concludes the platform holds."""

    counter: int
    epoch: int
    wrapped: bytes
    capsule: bytes
    slot: int
    rolled_back: bool  # an incomplete update was discarded
    previous: tuple[int, bytes, bytes] | None  # the superseded record, if kept


class Uninitialised(Exception):
    """No committed state at all. Distinct from a rollback."""


class CapsuleStore:
    """The commit protocol of Section `sec:design`, over a device.

    The whole design is one idea: **advancing the counter is the commit, and it
    is the only step that cannot be undone.** Before it, a crash leaves a slot
    that recovery discards because no counter names it. After it, every
    remaining step is idempotent and can simply be redone.
    """

    def __init__(self, device: Device) -> None:
        self.dev = device

    # -- reading -------------------------------------------------------------

    def _read_counter(self) -> tuple[int, int, int] | None:
        """The highest valid counter record, as `(value, slot, which record)`.

        Returning *which* record won matters: the next commit must write the
        other one. Alternating by epoch parity instead -- the obvious thing, and
        what this did first -- overwrites the only valid record every second
        commit, and a write torn there destroys the store outright. Experiment
        A2 found that; it is the reason this returns three values.
        """
        best = None
        for i, name in enumerate(COUNTERS):
            got = decode_counter(self.dev.read(name))
            if got and (best is None or got[0] > best[0]):
                best = (got[0], got[1], i)
        return best

    def recover(self) -> State:
        """The restart routine. Idempotent: running it twice changes nothing.

        Section `sec:design` says to compare the monotonic counter against the
        capsule's epoch and roll back an incomplete update. That is these three
        lines, and the reason it works is that the counter is advanced last.
        """
        ctr = self._read_counter()
        if ctr is None:
            raise Uninitialised("no valid counter record")
        value, _slot_hint, _which = ctr

        parsed = {i: decode_slot(self.dev.read(name))
                  for i, name in enumerate(SLOTS)}
        committed = [(i, r) for i, r in parsed.items()
                     if r is not None and r[0] == value]
        if len(committed) != 1:
            raise Uninitialised(
                f"counter names epoch {value}; {len(committed)} slots hold it")
        idx, (epoch, wrapped, capsule) = committed[0]

        # Anything claiming a later epoch than the counter is an update that
        # never committed. It is discarded, not honoured: that is the rollback.
        ahead = [i for i, r in parsed.items() if r is not None and r[0] > value]
        # Anything earlier is the superseded record, kept until the erase step
        # completes. Section `sec:impl` invariant 2 is about this one.
        behind = [(i, r) for i, r in parsed.items()
                  if r is not None and r[0] < value]

        return State(counter=value, epoch=epoch, wrapped=wrapped,
                     capsule=capsule, slot=idx, rolled_back=bool(ahead),
                     previous=behind[0][1] if behind else None)

    # -- writing -------------------------------------------------------------

    def provision(self, wrapped: bytes, capsule: bytes) -> None:
        """The first record. Written to slot 0 with epoch 1."""
        self.dev.write(SLOTS[0], encode_slot(1, wrapped, capsule))
        self.dev.sync(SLOTS[0])
        self.dev.write(COUNTERS[0], encode_counter(1, 0))
        self.dev.sync(COUNTERS[0])

    def commit(self, wrapped: bytes, capsule: bytes, *,
               skip_syncs: frozenset[int] = frozenset()) -> None:
        """Re-arm: write the new epoch, then commit it, then erase the old.

        The order is the claim. Each device operation is a fault-injection
        point, and experiments A1-A4 walk every one.

        `skip_syncs` names syncs to omit, by their 1-based position among the
        syncs of this commit. It exists for experiment A3: a sync that can be
        removed without breaking anything was not doing anything, and the only
        way to find out is to remove it.
        """
        state = self.recover()
        ctr = self._read_counter()
        assert ctr is not None
        new_epoch = state.counter + 1
        target = 1 - state.slot                 # the inactive slot
        ctr_target = 1 - ctr[2]                 # the older counter record
        n_sync = 0

        def maybe_sync(name: str) -> None:
            nonlocal n_sync
            n_sync += 1
            if n_sync not in skip_syncs:
                self.dev.sync(name)

        # 1-2. The new record reaches the medium.
        self.dev.write(SLOTS[target], encode_slot(new_epoch, wrapped, capsule))
        maybe_sync(SLOTS[target])

        # 3. "and its digest confirmed" -- read it back rather than trusting the
        #    write. A medium that accepted a torn write and reported success is
        #    caught here, before anything irreversible happens.
        check = decode_slot(self.dev.read(SLOTS[target]))
        if check is None or check[0] != new_epoch:
            raise IOError("new capsule failed verification after write")

        # 4-5. THE COMMIT POINT. Before this line a crash rolls back; after it,
        #      a crash rolls forward. There is no instant at which both or
        #      neither slot is named. The record written is the *older* of the
        #      two counter records, so the one currently in force survives any
        #      failure of this write.
        self.dev.write(COUNTERS[ctr_target], encode_counter(new_epoch, target))
        maybe_sync(COUNTERS[ctr_target])

        # 6-7. Erase the superseded record. Idempotent, and a crash here leaves
        #      the store correct but with a stale slot that the next recovery
        #      ignores and the next commit overwrites.
        self.dev.write(SLOTS[state.slot], b"")
        maybe_sync(SLOTS[state.slot])


#: The number of device operations one `commit` performs, when nothing fails.
#: Derived rather than asserted, so that adding a step to the protocol cannot
#: silently leave part of it unexercised.
def commit_operations(wrapped: bytes, capsule: bytes) -> int:
    dev = SimDevice()
    store = CapsuleStore(dev)
    store.provision(wrapped, capsule)
    before = dev.ops
    store.commit(wrapped, capsule)
    return dev.ops - before


def fault_points(wrapped: bytes, capsule: bytes) -> Iterator[int]:
    """Every operation index inside a commit, 1-based relative to that commit."""
    return iter(range(1, commit_operations(wrapped, capsule) + 1))
