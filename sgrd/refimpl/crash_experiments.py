"""Experiments A1-A4: is the capsule commit crash-consistent?

Section `sec:design` requires the capsule commit and the key update to be
atomic, and says that at whatever instant power is lost the platform holds
exactly one coherent (key, capsule) pair. Section `sec:impl` turns that into
three invariants to check after a restart. These experiments check them.

**What is tested and what is not.** The claim splits in two. The *algorithm* --
the order of writes and syncs, and the rollback rule at restart -- is settled
here, exhaustively: every device operation in the commit sequence is a fault
point, and every fault point is cut, torn, and de-synced. The *medium* is not.
Whether a write that returned actually reached the platters, whether a sector
can tear, whether a flash translation layer reorders: none of that is visible
from user space. `store.MEDIUM_REQUIREMENTS` states what the algorithm needs of
the medium so that a datasheet can be checked against it.

Four experiments, each answering a different question.

=====  =====================================================================
A1     Cut power before every operation. Does a restart always find exactly
       one coherent pair?
A2     Tear a write at every byte boundary that matters. Is a partial record
       always rejected rather than half-believed?
A3     Skip each sync in turn and drop what was not durable. **Which syncs
       are load-bearing?** This is the ablation; a sync that can be removed
       without breaking anything was not doing anything.
A4     Kill a real process writing real files. Does the algorithm behave the
       way the simulation says?
=====  =====================================================================

A3 is the one that cannot be done by killing a process: the kernel outlives the
writer and flushes its page cache regardless. Only a simulated device can
decline to.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Sequence

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from . import store as st
from . import tpre
from .experiments import Check

VESSEL = b"IMO9074729"
N_EPOCHS = 8


@dataclass(frozen=True)
class Epochs:
    """Real capsules and wrapped keys for a run of epochs.

    Built once with the actual primitives, so that "the pair is coherent" means
    the capsule really carries the key that really unwraps the working keys, not
    that two byte strings were filed together.
    """

    authority: tpre.RecoveryAuthority
    working_keys: bytes
    keks: dict[int, bytes]
    wrapped: dict[int, tuple[bytes, bytes]]
    capsules: dict[int, bytes]

    def blob(self, epoch: int) -> tuple[bytes, bytes]:
        nonce, ct = self.wrapped[epoch]
        return nonce + ct, self.capsules[epoch]

    def coherent(self, wrapped: bytes, capsule: bytes) -> int | None:
        """Which epoch is this pair, if it is a valid pair at all?

        Returns the epoch, or `None` when the two halves belong to different
        epochs or either is not a record this run ever produced. A store that
        handed back slot A's key with slot B's capsule would be found here and
        nowhere else.
        """
        for epoch, cap in self.capsules.items():
            if cap != capsule:
                continue
            kek = self.keks[epoch]
            nonce, ct = wrapped[:12], wrapped[12:]
            try:
                if AESGCM(kek).decrypt(nonce, ct, VESSEL) == self.working_keys:
                    return epoch
            except InvalidTag:
                return None
            return None
        return None


def build_epochs(n: int = N_EPOCHS) -> Epochs:
    authority, _shares = tpre.provision(2, 3)
    kw = os.urandom(64)
    keks, wrapped, capsules = {}, {}, {}
    for e in range(1, n + 1):
        kek = os.urandom(32)
        nonce = os.urandom(12)
        keks[e] = kek
        wrapped[e] = (nonce, AESGCM(kek).encrypt(nonce, kw, VESSEL))
        capsules[e] = _serialise(
            tpre.seal(authority, kek, vessel_id=VESSEL, epoch=e))
    return Epochs(authority, kw, keks, wrapped, capsules)


def _serialise(cap: tpre.Capsule) -> bytes:
    from . import group as gp
    return (gp.i2osp(cap.c1) + cap.nonce + len(cap.ct).to_bytes(4, "big")
            + cap.ct + cap.epoch.to_bytes(8, "big"))


# --- the invariants ----------------------------------------------------------

def check_invariants(state: st.State, ep: Epochs, *, counter_before: int,
                     epoch_in_flight: int) -> list[str]:
    """The three invariants of Section `sec:impl`, plus two the design implies.

    Returns the names of any that failed. An empty list is a pass.
    """
    bad = []
    # I1 -- exactly one coherent pair. `recover` raising `Uninitialised` is the
    # other way this fails, and the caller turns that into a failure too.
    found = ep.coherent(state.wrapped, state.capsule)
    if found is None:
        bad.append("I1 pair is not coherent")
    elif found != state.epoch:
        bad.append(f"I1 slot says epoch {state.epoch}, contents say {found}")

    # I2 -- the counter names the epoch that is actually there.
    if state.counter != state.epoch:
        bad.append(f"I2 counter {state.counter} != epoch {state.epoch}")

    # I3 -- monotonicity. The counter may not go backwards across a crash.
    if state.counter < counter_before:
        bad.append(f"I3 counter went back {counter_before} -> {state.counter}")

    # I4 -- the outcome is one of exactly two possibilities. A crash during an
    # update either rolls back to the previous epoch or forward to the new one;
    # there is no third answer, and in particular no epoch that was never the
    # subject of a commit.
    if state.counter not in (counter_before, epoch_in_flight):
        bad.append(f"I4 counter {state.counter} is neither "
                   f"{counter_before} nor {epoch_in_flight}")

    # I5 -- the rollback flag agrees with what happened.
    if state.rolled_back and state.counter != counter_before:
        bad.append("I5 reported a rollback but moved forward anyway")
    return bad


def _replay(ep: Epochs, fault: st.Fault, fault_epoch: int,
            survive: frozenset[str], skip_syncs: frozenset[int]
            ) -> tuple[st.SimDevice, int]:
    """Run provision plus commits, arming the fault only for one commit.

    Arming per-commit rather than counting from the beginning keeps the fault
    index meaning "the k-th operation of *this* commit", which is what the
    experiment is about and what stays stable if the number of prior commits
    changes.
    """
    dev = st.SimDevice()
    dev.survive = survive
    store = st.CapsuleStore(dev)
    store.provision(*ep.blob(1))
    for e in range(2, fault_epoch + 1):
        if e == fault_epoch:
            dev.fault = st.Fault(dev.ops + fault.at, fault.mode, fault.prefix)
        try:
            store.commit(*ep.blob(e), skip_syncs=skip_syncs)
        except st.PowerLoss:
            return dev, e
        finally:
            dev.fault = None
    return dev, fault_epoch


#: Everything the protocol can have in flight. Enumerating subsets of this is
#: how reordering is covered: any of them may or may not have reached the
#: medium when power went.
IN_FLIGHT = tuple(st.SLOTS) + tuple(st.COUNTERS)


def _subsets(names: Sequence[str]):
    for mask in range(1 << len(names)):
        yield frozenset(n for i, n in enumerate(names) if mask >> i & 1)


def _one_trial(ep: Epochs, at: int, mode: str, prefix: int, fault_epoch: int,
               survive: frozenset[str] = frozenset(),
               skip_syncs: frozenset[int] = frozenset()
               ) -> tuple[str, list[str]]:
    dev, in_flight = _replay(ep, st.Fault(at, mode, prefix), fault_epoch,
                             survive, skip_syncs)
    store = st.CapsuleStore(dev.snapshot())
    try:
        state = store.recover()
    except st.Uninitialised as exc:
        return "uninitialised", [f"I1 {exc}"]
    bad = check_invariants(state, ep, counter_before=fault_epoch - 1,
                           epoch_in_flight=in_flight)
    # I6 -- recovery is idempotent. A second restart must reach the same place.
    again = store.recover()
    if (again.counter, again.epoch, again.wrapped) != \
            (state.counter, state.epoch, state.wrapped):
        bad.append("I6 recovery is not idempotent")
    return ("rolled back" if state.counter == fault_epoch - 1
            else "rolled forward"), bad


# --- A1: cut power before every operation ------------------------------------

def a1_power_cut(ep: Epochs) -> list[Check]:
    """Every operation of a commit, cut at, from both slot parities.

    Parity matters: the protocol alternates slots and counter records, so a bug
    that only bites when the active slot is 1 would hide behind a sweep that
    only ever crashed on the first commit.
    """
    out: list[Check] = []
    n_ops = st.commit_operations(*ep.blob(1))
    back = fwd = trials = 0
    failures: list[str] = []
    for fault_epoch in (2, 3, 4, 5):
        for at in range(1, n_ops + 1):
            for survive in _subsets(IN_FLIGHT):
                trials += 1
                outcome, bad = _one_trial(ep, at, "cut", 0, fault_epoch,
                                          survive)
                if bad:
                    failures.append(f"epoch {fault_epoch} op {at} "
                                    f"survive={sorted(survive)}: "
                                    f"{'; '.join(bad)}")
                elif outcome == "rolled back":
                    back += 1
                else:
                    fwd += 1
    out.append(Check(
        "A", f"A1 power cut before each of {n_ops} operations, "
             f"x4 epochs x {1 << len(IN_FLIGHT)} survival subsets",
        f"{trials} coherent", f"{trials - len(failures)} coherent",
        not failures,
        f"{back} rolled back to the previous epoch, {fwd} rolled forward to "
        "the new one; no other outcome occurred. The subsets cover reordering: "
        "any pending write may or may not have reached the medium",
        "; ".join(failures[:3])))
    out.append(Check(
        "A", "A1 both outcomes actually occur", "both",
        "both" if back and fwd else ("only rollback" if back else "only roll-forward"),
        bool(back and fwd),
        "a protocol that only ever rolled back would pass the invariants and "
        "lose every update"))
    return out


# --- A2: tear a write --------------------------------------------------------

def a2_torn_writes(ep: Epochs) -> list[Check]:
    """Partial records, at the boundaries where a naive parser would be fooled.

    The digest is at the end, so the interesting prefixes are the ones that
    carry a plausible header: one byte, the magic alone, everything but the
    digest, and everything but one byte of it.
    """
    out: list[Check] = []
    n_ops = st.commit_operations(*ep.blob(1))
    record = st.encode_slot(3, *ep.blob(3))
    prefixes = [0, 1, 8, 16, len(record) // 2,
                len(record) - 33, len(record) - 32, len(record) - 1]
    failures: list[str] = []
    trials = 0
    for fault_epoch in (2, 3):
        for at in range(1, n_ops + 1):
            for prefix in prefixes:
                trials += 1
                _outcome, bad = _one_trial(ep, at, "tear", prefix, fault_epoch)
                if bad:
                    failures.append(f"epoch {fault_epoch} op {at} "
                                    f"prefix {prefix}: {'; '.join(bad)}")
    out.append(Check(
        "A", f"A2 torn writes at {len(prefixes)} boundaries x {n_ops} operations",
        f"{trials} coherent", f"{trials - len(failures)} coherent",
        not failures,
        "including a record complete but for its last digest byte",
        "; ".join(failures[:3])))

    # A record whose digest is right but whose length fields are wrong must also
    # fail: a torn write that happened to stop at a plausible boundary would
    # otherwise be re-digested by an attacker and accepted.
    body = record[:-32]
    forged = body[:20] + hashlib.sha256(body[:20]).digest()
    rejected = st.decode_slot(forged) is None
    out.append(Check(
        "A", "A2 a record with a valid digest over a truncated body",
        "rejected", "rejected" if rejected else "accepted", rejected,
        "the magic and the digest agree; the length fields do not"))
    return out


# --- A3: which syncs are load-bearing? --------------------------------------

def a3_sync_ablation(ep: Epochs) -> list[Check]:
    """Remove each sync in turn and see what breaks.

    This is the experiment that says whether the protocol's syncs are doing
    anything. A sync whose removal changes nothing is either redundant or is
    covering a case this sweep does not reach; either way it should be known
    about rather than carried on faith.

    A killed process cannot produce this: the kernel writes its page cache out
    regardless. Only a device that declines to make data durable can.
    """
    out: list[Check] = []
    n_ops = st.commit_operations(*ep.blob(1))

    # Name the syncs by what they sync, so the finding reads as a statement
    # about the protocol rather than about an index.
    dev = st.SimDevice()
    store = st.CapsuleStore(dev)
    store.provision(*ep.blob(1))
    before = dev.ops
    store.commit(*ep.blob(2))
    syncs = [t.split(":", 1)[1] for t in dev.trace[before:]
             if t.split(":", 1)[1].startswith("sync")]
    names = ["the new slot record", "the counter", "the erase of the old slot"]

    n_load = 0
    for k, label in enumerate(names, start=1):
        broke: list[str] = []
        for fault_epoch in (2, 3, 4, 5):
            for at in range(1, n_ops + 1):
                for survive in _subsets(IN_FLIGHT):
                    _outcome, bad = _one_trial(
                        ep, at, "cut", 0, fault_epoch, survive,
                        skip_syncs=frozenset({k}))
                    broke.extend(bad)
        load = bool(broke)
        n_load += load
        out.append(Check(
            "A", f"A3 sync {k} of 3 removed, on {label}",
            "load-bearing" if load else "no effect in this sweep",
            "load-bearing" if load else "no effect in this sweep",
            True,
            (f"{len(broke)} invariant failures without it. First: {broke[0]}")
            if load else
            "removing it breaks nothing under any cut or survival subset, so "
            "it is redundant for correctness -- it still bounds how much work "
            "a restart has to redo",
            ""))
    out.append(Check(
        "A", "A3 at least one sync is load-bearing", "yes",
        "yes" if n_load else "no", bool(n_load),
        f"{n_load} of {len(names)} syncs, when removed, break an invariant "
        "under some cut. The others are not doing correctness work, and "
        "saying so is the point of the ablation"))
    return out


# --- A4: kill a real process -------------------------------------------------

_CHILD = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
from sgrd.refimpl import store as st
cfg = json.load(open(sys.argv[2], 'rb'))
blobs = [(bytes.fromhex(w), bytes.fromhex(c)) for w, c in cfg['blobs']]
dev = st.RealDevice(cfg['root'])
store = st.CapsuleStore(dev)
store.provision(*blobs[0])
for i, blob in enumerate(blobs[1:], start=2):
    if i == cfg['fault_epoch']:
        dev.fault = st.Fault(dev.ops + cfg['at'], cfg['mode'], cfg['prefix'])
    store.commit(*blob)
    dev.fault = None
print('survived')
"""


def a4_real_process_kill(ep: Epochs, repeats: int = 1) -> list[Check]:
    """The same sweep against real files, with the writer actually dying.

    `os._exit` bypasses interpreter cleanup, so nothing sits in a Python buffer.
    What it cannot bypass is the kernel, which is exactly why A3 exists.
    """
    out: list[Check] = []
    root_pkg = str(_package_root())
    n_ops = st.commit_operations(*ep.blob(1))
    child = os.path.join(tempfile.gettempdir(), "sgrd_crash_child.py")
    with open(child, "w", encoding="utf-8") as fh:
        fh.write(_CHILD)

    failures: list[str] = []
    mismatches: list[str] = []
    trials = killed = 0
    for fault_epoch in (2, 3):
        for at in range(1, n_ops + 1):
            # What the simulation says should happen. A killed process is the
            # case where everything already written reaches the medium anyway:
            # the kernel outlives the writer and flushes its page cache. So the
            # matching simulated cut is the one where all pending writes survive.
            predicted, _ = _one_trial(ep, at, "cut", 0, fault_epoch,
                                      frozenset(IN_FLIGHT))
            for _ in range(repeats):
                trials += 1
                with tempfile.TemporaryDirectory() as tmp:
                    root = os.path.join(tmp, "store")
                    cfg = os.path.join(tmp, "cfg.json")
                    with open(cfg, "w", encoding="utf-8") as fh:
                        json.dump({
                            "root": root, "at": at, "mode": "cut", "prefix": 0,
                            "fault_epoch": fault_epoch,
                            "blobs": [[w.hex(), c.hex()]
                                      for w, c in (ep.blob(e)
                                                   for e in range(1, 4))],
                        }, fh)
                    proc = subprocess.run(
                        [sys.executable, child, root_pkg, cfg],
                        capture_output=True, timeout=60)
                    if proc.returncode != 0:
                        killed += 1
                    state = _recover_real(root)
                    if state is None:
                        failures.append(f"epoch {fault_epoch} op {at}: "
                                        "no committed state after restart")
                        continue
                    bad = check_invariants(
                        state, ep, counter_before=fault_epoch - 1,
                        epoch_in_flight=fault_epoch)
                    if bad:
                        failures.append(f"epoch {fault_epoch} op {at}: "
                                        f"{'; '.join(bad)}")
                    observed = ("rolled back"
                                if state.counter == fault_epoch - 1
                                else "rolled forward")
                    if observed != predicted:
                        mismatches.append(
                            f"epoch {fault_epoch} op {at}: simulation said "
                            f"{predicted}, real run {observed}")
    out.append(Check(
        "A", f"A4 real process killed at each of {n_ops} operations",
        f"{trials} coherent", f"{trials - len(failures)} coherent",
        not failures,
        f"{killed} of {trials} children died before returning. Real files, "
        "real os.write and os.fsync, os._exit at the fault point",
        "; ".join(failures[:3])))
    out.append(Check(
        "A", "A4 the real runs match what the simulation predicted",
        f"{trials} agree", f"{trials - len(mismatches)} agree",
        not mismatches,
        "the simulated counterpart of a killed process is the cut in which "
        "every pending write survives, since the kernel flushes its page cache "
        "regardless of what happened to the writer. Agreement is what licenses "
        "reading A1-A3, which are simulated, as statements about the algorithm",
        "; ".join(mismatches[:3])))
    return out


def _recover_real(root: str) -> st.State | None:
    try:
        return st.CapsuleStore(st.RealDevice(root)).recover()
    except st.Uninitialised:
        return None


def _package_root():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent.parent


# --- driver ------------------------------------------------------------------

CRASH_EXPERIMENTS = {
    "A1": ("Power cut before every operation of the commit", a1_power_cut),
    "A2": ("Torn writes at the boundaries that matter", a2_torn_writes),
    "A3": ("Which syncs are load-bearing, by ablation", a3_sync_ablation),
    "A4": ("A real process killed while writing real files",
           a4_real_process_kill),
}


def run_all(ep: Epochs | None = None) -> list[Check]:
    ep = ep or build_epochs()
    checks: list[Check] = []
    for _, fn in CRASH_EXPERIMENTS.values():
        checks.extend(fn(ep))
    return checks
