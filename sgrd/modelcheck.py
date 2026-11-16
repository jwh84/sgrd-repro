"""Exhaustive model checking of the restoration controller (Sections
`sec:method`, `sec:results`, Appendix `app:props`).

The controller and its environment form a finite transition system. We build the
full reachable state set by BFS, then check the ten properties of Table
`tab:props` over the reachable *transition* relation -- not over states. The
distinction is the one the paper insists on: a state in which the vessel has
been restored and is subsequently seized again is perfectly legitimate, whereas
a *transition* into the restored phase from an unattested configuration is not.

Two variants share one environment:

* **B0** -- on-board master recovery secret; restoration authority concentrated
  in one organisation, so one coerced holder suffices; single-step restoration
  on presentation of any authority; sealed key material retained on board; no
  post-possession attestation; no monotonic counter.
* **H1** -- SGRD as specified in Sections `sec:design` and `sec:staged`.

Revision for the final defence (review items M1, M2, M4, M8):

* P4-P6 are checked over every stage-advancing transition, not only the final
  transition into RESTORED, so "every stage is gated on Safe", "the stages are
  never skipped" and "nothing past R1 runs unattested" are what is verified.
* A spurious denial -- denial without possession, a false trigger -- is an
  environment move (M4).
* The offline path of Algorithm 2 is in the model, with the adversary's moves
  against it: stealing the offline credential, stealing a custody attestation,
  presenting both while in possession, and extracting the transport key sk_tr
  from the module (M8). How much theft is allowed is a setting, so the residual
  risks can be reported separately from the main result.
* Three designs taken from key-management practice -- HSM quorum approval (K1),
  attestation-gated key release (K2) and key escrow (K3) -- are variants (M2).

The defaults of :data:`SETTINGS` are the configuration the thesis reports as its
main result. :func:`verify` takes another :class:`Settings` for the others.

Note on state counts: the size of the reachable relation depends on the
encoding, not only on the semantics. The counts this module reports are
therefore compared against the published ones for information; the load-bearing
result is the satisfied/violated verdict per property, which is encoding-robust.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator, NamedTuple

# --- Phases ------------------------------------------------------------------

NORMAL, DENIED, RESTORED = "NORMAL", "DENIED", "RESTORED"
R0, R1, R2, R3 = "R0", "R1", "R2", "R3"
STAGES = (R0, R1, R2, R3)

# --- Authority sources -------------------------------------------------------

NONE, LEGIT, COERCED, ONBOARD = "none", "legitimate", "coerced", "on-board"
# Algorithm 2: the boarding party's offline credential, and the same credential
# presented by an adversary who holds a stolen custody attestation as well.
OFFLINE, OFFLINE_STOLEN = "offline", "offline-stolen"

# --- Command authority over the navigation kernel ----------------------------
#
# Section `sec:kernel` makes this single-valued: the ROC holds it in normal
# operation, it reverts to AUTONOMOUS when denial fires -- the kernel holds a
# minimum-risk condition without being commanded -- a boarding party may take it
# on custody evidence, and R2 returns it to the ROC. No transition assigns it to
# the adversary, and P12 is the claim that none can.

ROC, BOARDING, AUTONOMOUS = "roc", "boarding-party", "autonomous"


class State(NamedTuple):
    phase: str
    possessed: bool  # adversary has uninterrupted physical access
    tampered: bool  # software/models/config altered during possession
    attested: bool  # measured state matches a post-possession baseline
    keys: bool  # K_w present or derivable on board
    extracted: bool  # A_cpi has been extracted
    auth: str  # source of the restoration authority now presented
    fresh: bool  # that authority is not stale
    safe: bool  # navigational safe-state predicate holds
    used: bool  # the authorisation has already been consumed
    denied: bool  # a denial episode is active
    nav_available: bool  # the navigation kernel is functioning
    command_owner: str  # who directs that kernel
    sensor_trusted: bool  # sensing and positioning revalidated since possession
    # Offline path (Algorithm 2). All False in a model without it.
    off_stolen: bool = False  # the adversary holds this epoch's offline credential
    cust_stolen: bool = False  # ... and a custody attestation still within validity
    sktr: bool = False  # the transport key sk_tr has been extracted from the module
    off_spent: bool = False  # this epoch's offline credential has been consumed


INITIAL = State(
    phase=NORMAL, possessed=False, tampered=False, attested=True, keys=True,
    extracted=False, auth=NONE, fresh=False, safe=True, used=False, denied=False,
    nav_available=True, command_owner=ROC, sensor_trusted=True,
)


@dataclass(frozen=True)
class Settings:
    """Which environment the controller is checked against.

    ``theft`` is how much of the offline path the adversary may steal: ``none``,
    ``one`` (the credential or the custody attestation, not both -- they come
    from disjoint issuers), or ``both``. ``sktr_extractable`` lets the adversary
    pull the transport key out of the module by invasive work, which is the
    residual path of review item M8 and is rated in Table tab:ap rather than
    assumed away.
    """

    stagewise: bool = True
    false_trigger: bool = True
    offline: bool = True
    theft: str = "none"
    sktr_extractable: bool = False


SETTINGS = Settings()
#: The settings under which the first version of this module was published.
SETTINGS_V1 = Settings(stagewise=False, false_trigger=False, offline=False)


def replace(s: State, **changes: object) -> State:
    """Copy a state with some fields changed.

    States are NamedTuples -- hashable, so they can be BFS keys, and cheap to
    compare, which matters when the reachable relation runs to thousands of
    edges. ``dataclasses.replace`` does not work on them; this wraps the
    documented ``_replace`` so the transition rules read as substitutions.
    """
    return s._replace(**changes)


@dataclass(frozen=True)
class Variant:
    """The two controller designs, as the handful of flags that differ."""

    name: str
    erase_keys_on_denial: bool  # H1 erases K_w and the helper datum
    onboard_recovery_secret: bool  # B0 can mint authority from on board
    monotonic_counter: bool  # H1 gives each episode a fresh token space
    staged: bool  # H1 walks R0->R1->R2->R3
    gate_custody: bool  # R0 requires custody regained
    gate_safe: bool  # every stage requires the safe-state predicate
    gate_attested: bool  # capability requires post-possession attestation
    gate_unused: bool  # the token must not have been consumed
    require_threshold: bool  # below-threshold (coerced) authority is refused
    nav_kernel_outside_denial: bool  # denial does not reach the navigation kernel
    offline_path: bool = False  # Algorithm 2 exists for this design


B0 = Variant(
    name="B0",
    erase_keys_on_denial=False,
    onboard_recovery_secret=True,
    monotonic_counter=False,
    staged=False,
    gate_custody=False,
    gate_safe=False,
    gate_attested=False,
    gate_unused=False,
    require_threshold=False,
    nav_kernel_outside_denial=False,
)

H1 = Variant(
    name="H1",
    erase_keys_on_denial=True,
    onboard_recovery_secret=False,
    monotonic_counter=True,
    staged=True,
    gate_custody=True,
    gate_safe=True,
    gate_attested=True,
    gate_unused=True,
    require_threshold=True,
    nav_kernel_outside_denial=True,
    offline_path=True,
)

# Three intermediate designs, so that the comparison is not only against a
# deliberately weak baseline. Each isolates one of the paper's contributions:
# B1 moves recovery authority off the platform and stages the return but leaves
# it concentrated in one organisation; B2 distributes the authority but hands
# everything back in a single step; B3 has both but no safe-state gate. Reading
# the three columns against H1 is what separates the contributions -- a 10/10
# against a 1/10 baseline does not.

B1 = Variant(  # off-board but centralised authority, staged restoration
    name="B1", erase_keys_on_denial=True, onboard_recovery_secret=False,
    monotonic_counter=True, staged=True, gate_custody=True, gate_safe=True,
    gate_attested=True, gate_unused=True, require_threshold=False,
    nav_kernel_outside_denial=True, offline_path=True,
)

B2 = Variant(  # threshold authority, single-step restoration
    name="B2", erase_keys_on_denial=True, onboard_recovery_secret=False,
    monotonic_counter=True, staged=False, gate_custody=True, gate_safe=True,
    gate_attested=True, gate_unused=True, require_threshold=True,
    nav_kernel_outside_denial=True, offline_path=True,
)

B3 = Variant(  # crypto-erasure and threshold authority, no safe-state gate
    name="B3", erase_keys_on_denial=True, onboard_recovery_secret=False,
    monotonic_counter=True, staged=True, gate_custody=True, gate_safe=False,
    gate_attested=True, gate_unused=True, require_threshold=True,
    nav_kernel_outside_denial=True, offline_path=True,
)

# Designs taken from key-management practice (Section sec:other, review item
# M2). Each keeps H1's crypto-erasure, the absence of an on-board secret and the
# kernel outside the denial boundary, and replaces only the restoration
# mechanism, so that what separates them from H1 is the mechanism.
_PRACTICE = dict(erase_keys_on_denial=True, onboard_recovery_secret=False,
                 nav_kernel_outside_denial=True, monotonic_counter=False,
                 gate_custody=False, gate_safe=False, gate_unused=False)
K1 = Variant(name="K1", staged=False, gate_attested=False, require_threshold=True,
             **_PRACTICE)  # HSM M-of-N quorum approval: threshold, nothing else
K2 = Variant(name="K2", staged=True, gate_attested=True, require_threshold=False,
             **_PRACTICE)  # attestation-gated key release: one broker, attestation
                           # (H1's staged controller is reused to carry the R1
                           # attestation, which favours K2)
K3 = Variant(name="K3", staged=False, gate_attested=False, require_threshold=False,
             **_PRACTICE)  # key escrow: one agent, no condition

VARIANTS = {"B0": B0, "K1": K1, "K2": K2, "K3": K3,
            "B1": B1, "B2": B2, "B3": B3, "H1": H1}
DESIGNS = ("B0", "K1", "K2", "K3", "B1", "B2", "B3", "H1")


# --- Transition relation -----------------------------------------------------

#: Code mutations used only by :func:`mutation_test`. Each names one rule of the
#: relation below and switches it off, so a property can be shown to depend on it.
MUTATIONS = ("no_cmd_revert", "no_reissue", "no_r1_check", "residue")


def successors(s: State, v: Variant, cfg: Settings = SETTINGS,
               mut: frozenset = frozenset()) -> Iterator[tuple[str, State]]:
    """All transitions enabled in ``s``, as ``(label, successor)`` pairs.

    Environment moves come first, then the denial actuator, then the controller.
    Nothing here is scheduled: every enabled move is explored, which is what
    makes the check exhaustive rather than a simulation.
    """
    off = cfg.offline and v.offline_path

    # --- Environment: the adversary and the world --------------------------

    if not s.possessed:
        # E1 seizure. Possession invalidates any attestation baseline, and it
        # aborts a restoration in progress -- the R0 custody conjunct again.
        # It also puts the sensing and positioning chain in doubt, which is why
        # R2 has to revalidate it before the vessel may go autonomous again.
        # Command reverts to the kernel: whoever held it -- the ROC over a link
        # the adversary can cut, or a boarding party now displaced -- no longer
        # does. Leaving it set was a modelling error that let P12 pass over 80
        # reachable states in which the adversary held the platform while
        # command was still assigned; exhaustive checking of the *states*, not
        # only the transitions, is what surfaced it.
        nxt = replace(s, possessed=True, attested=False, sensor_trusted=False,
                      command_owner=(s.command_owner if "no_cmd_revert" in mut
                                     else AUTONOMOUS))
        if s.phase in STAGES:
            nxt = replace(nxt, phase=DENIED)
        yield "seize", nxt
    else:
        yield "regain_custody", replace(s, possessed=False)

        if not s.tampered:
            # E3 modification in captivity (hazard 4 of Section sec:staged).
            yield "modify", replace(s, tampered=True, attested=False)

        if (s.keys or "residue" in mut) and not s.extracted:
            # E4 extraction. Requires the hardware *and* usable key material;
            # this is the transition H1 removes by erasing K_w at denial.
            yield "extract", replace(s, extracted=True)

        # M8: the transport key survives denial inside the module; pulling it
        # out is invasive work rated in Table tab:ap, allowed only if the
        # setting says so. With it and a stolen offline credential the
        # precomputed opening O decrypts to K_e without passing R0.
        if off and cfg.sktr_extractable and not s.sktr:
            yield "extract_sk_tr", replace(s, sktr=True)
        if off and s.sktr and s.off_stolen and not s.extracted:
            yield "decrypt_offline_opening", replace(s, extracted=True)

    # E5 the navigational situation changes in both directions.
    yield ("safe_lost" if s.safe else "safe_regained"), replace(s, safe=not s.safe)

    # E6 authorisation goes stale.
    if s.fresh:
        yield "authorisation_stale", replace(s, fresh=False)

    # E7 holders issue legitimate authority. M2 of Algorithm alg:proto has them
    # verify that custody has been regained before signing.
    #
    # The guard must admit *re-issuance* of an authority that has gone stale,
    # not only the first issuance. Each token carries counter c+1, so holders can
    # always mint a new one; blocking that (by guarding on `auth != LEGIT`) is a
    # modelling artefact and it manufactures a P9 violation. The mutation
    # "no_reissue" reinstates that artefact so that P9 can be seen to bite.
    if (s.denied and not s.possessed and not (s.auth == LEGIT and s.fresh)
            and not ("no_reissue" in mut and s.auth == LEGIT)):
        yield "issue_legitimate_authority", replace(s, auth=LEGIT, fresh=True)

    # E8 the adversary obtains below-threshold authority: a coerced or bribed
    # holder (class A4), which combined with possession is class A5.
    if s.denied and s.auth == NONE:
        yield "coerce_holder", replace(s, auth=COERCED, fresh=True)

    # E9 off-board remediation, possible only once custody is back.
    if s.tampered and not s.possessed:
        yield "off_board_remediation", replace(s, tampered=False)

    # E10 B0 only: the on-board master recovery secret is always to hand.
    if v.onboard_recovery_secret and s.denied and s.auth == NONE:
        yield "read_onboard_secret", replace(s, auth=ONBOARD, fresh=True)

    # E11 a boarding party takes command of the navigation kernel. Section
    # `sec:kernel` gates this on custody evidence, which is why the guard is
    # `not s.possessed` -- the same conjunct R0 uses. Nothing in the relation
    # assigns command to the adversary, and P12 asserts that no path does.
    if (s.denied and not s.possessed and s.nav_available
            and s.command_owner == AUTONOMOUS):
        yield "boarding_party_takes_command", replace(s, command_owner=BOARDING)

    # E12-E15 the offline path of Algorithm 2 (M8).
    if off:
        # The boarding party presents this epoch's credential together with a
        # custody attestation from the disjoint issuer.
        if (s.denied and not s.possessed and not s.off_spent
                and not s.off_stolen and s.auth != OFFLINE):
            yield "present_offline_credential", replace(s, auth=OFFLINE, fresh=True)
        if cfg.theft in ("one", "both"):
            both = cfg.theft == "both"
            if not s.off_stolen and (both or not s.cust_stolen):
                yield "steal_offline_credential", replace(s, off_stolen=True)
            if not s.cust_stolen and (both or not s.off_stolen):
                yield "steal_custody_proof", replace(s, cust_stolen=True)
        # Holding both, the adversary presents them while still in possession;
        # the platform has no way to tell, which is the residual risk Section
        # sec:proto states for the validity window of the attestation.
        if (s.possessed and s.denied and s.off_stolen and s.cust_stolen
                and not s.off_spent and s.auth == NONE):
            yield "present_stolen_offline", replace(s, auth=OFFLINE_STOLEN, fresh=True)

    # --- Denial actuator ----------------------------------------------------

    def deny(st: State) -> State:
        return replace(
            st, phase=DENIED, denied=True, auth=NONE, fresh=False,
            # H1 erases K_w and the helper datum; B0 retains sealed material.
            keys=not v.erase_keys_on_denial,
            # A monotonic counter opens a fresh token space per episode. Without
            # one, a previously consumed authorisation is still current.
            used=False if v.monotonic_counter else st.used,
            # The boundary of Section `sec:kernel`. Where the kernel lies
            # outside it, denial leaves the vessel navigable and the kernel
            # holds a minimum-risk condition without being commanded. Where it
            # does not -- B0's blunt lockout -- denial takes navigation with it,
            # which is what P11 detects.
            nav_available=v.nav_kernel_outside_denial,
            command_owner=AUTONOMOUS,
        )

    if s.phase == NORMAL and s.possessed and not s.denied:
        yield "deny", deny(s)
    # M4: denial fires without possession -- a pilot boarding, a port-state
    # inspection, a sensor fault. Restoration after it must satisfy the same
    # twelve properties.
    if cfg.false_trigger and s.phase == NORMAL and not s.possessed and not s.denied:
        yield "spurious_deny", deny(s)

    # --- Restoration controller --------------------------------------------

    if v.staged:
        yield from _staged_controller(s, v, mut)
    else:
        yield from _single_step_controller(s, v)

    # Normal operation resumes after a restoration, so a later seizure can fire
    # denial again. Without this the model could not express repeated episodes,
    # and P9 would be checked over a single episode only.
    if s.phase == RESTORED:
        yield "resume_normal_operation", replace(s, phase=NORMAL)


def _custody_ok(s: State, v: Variant) -> bool:
    """What the platform can know of custody: that it is back, or that a valid
    attestation says so -- which a stolen one does while it is still valid."""
    return (not v.gate_custody) or (not s.possessed) or s.auth == OFFLINE_STOLEN


def _gate(s: State, v: Variant) -> bool:
    """The conjuncts common to every stage of the hardened controller."""
    return (not v.gate_safe or s.safe) and _custody_ok(s, v)


def _authority_ok(s: State, v: Variant) -> bool:
    if v.require_threshold:
        # The offline credential carries a threshold signature issued in advance.
        return s.auth in (LEGIT, OFFLINE, OFFLINE_STOLEN)
    return s.auth != NONE


#: Restoration advances the epoch, which invalidates every unused offline
#: credential, including a stolen one (Section sec:proto).
_EPOCH_ADVANCE = dict(off_spent=False, off_stolen=False)


def _staged_controller(s: State, v: Variant,
                       mut: frozenset = frozenset()) -> Iterator[tuple[str, State]]:
    if s.phase == DENIED:
        # R0 admission: Auth AND Safe AND NOT Possessed AND NOT Used.
        if (_authority_ok(s, v) and s.fresh and _gate(s, v)
                and (not v.gate_unused or not s.used)):
            nxt = replace(s, phase=R0)
            if s.auth in (OFFLINE, OFFLINE_STOLEN):
                nxt = replace(nxt, off_spent=True)  # one-time n_cust, counter
            yield "R0_admit", nxt

    elif s.phase == R0:
        if _gate(s, v):
            yield "R0_to_R1", replace(s, phase=R1)

    elif s.phase == R1:
        # Re-attestation. Nothing from A_cpi executes in this stage.
        if s.tampered and "no_r1_check" not in mut:
            yield "R1_attestation_mismatch", replace(s, phase=DENIED)
        elif _gate(s, v):
            yield "R1_attest_ok", replace(s, phase=R2, attested=True)

    elif s.phase == R2:
        # Trusted degraded navigation. R2 revalidates the sensing and
        # positioning chain and returns command from the boarding party to the
        # ROC, and only then may the vessel proceed toward autonomy.
        if _gate(s, v) and (not v.gate_attested or s.attested):
            yield "R2_to_R3", replace(
                s, phase=R3, sensor_trusted=True, command_owner=ROC)

    elif s.phase == R3:
        # Autonomy resumes only on a revalidated sensing chain: the safe-state
        # predicate the later stages are gated on is worth nothing if the
        # evidence behind it is the adversary's.
        if _gate(s, v) and (not v.gate_attested or s.attested) and s.sensor_trusted:
            # Only here is K_w re-derived and the counter advanced.
            yield "R3_restore", replace(
                s, phase=RESTORED, keys=True, used=True, denied=False,
                auth=NONE, fresh=False, **_EPOCH_ADVANCE,
            )

    # Abort to denial rather than hold a partially restored vessel in an unsafe
    # situation. Intended behaviour, not a failure mode.
    if s.phase in STAGES and v.gate_safe and not s.safe:
        yield "abort_unsafe", replace(s, phase=DENIED)


def _single_step_controller(s: State, v: Variant) -> Iterator[tuple[str, State]]:
    # Present authority, get everything back at once: no ordering, no
    # attestation before execution, no per-stage safe-state re-evaluation.
    # B2 shows what distributing the authority buys when the return is still a
    # single step, which is why the threshold conjunct is honoured here too.
    #
    # It keeps every conjunct a single step *can* hold -- custody, safe state,
    # unspent token -- so that what B2 lacks against H1 is only what ordering
    # itself provides. Attestation is not among them: `attested` is established
    # by R1, so requiring it here would be staging under another name.
    if s.phase == DENIED:
        if (_authority_ok(s, v) and _gate(s, v)
                and (not v.gate_unused or not s.used)):
            yield "restore", replace(
                s, phase=RESTORED, keys=True, used=True, denied=False,
                sensor_trusted=True, command_owner=ROC, **_EPOCH_ADVANCE,
            )


# --- Reachability ------------------------------------------------------------

class Relation(NamedTuple):
    states: set[State]
    transitions: list[tuple[State, str, State]]


def explore(v: Variant, initial: State = INITIAL, cfg: Settings = SETTINGS,
            mut: frozenset = frozenset()) -> Relation:
    """BFS the full reachable state set and collect every reachable transition."""
    seen = {initial}
    queue = deque([initial])
    edges: list[tuple[State, str, State]] = []
    while queue:
        s = queue.popleft()
        for label, t in successors(s, v, cfg, mut):
            edges.append((s, label, t))
            if t not in seen:
                seen.add(t)
                queue.append(t)
    return Relation(seen, edges)


# --- Properties (Appendix app:props) ----------------------------------------

_ORDER = {DENIED: 0, R0: 1, R1: 2, R2: 3, R3: 4, RESTORED: 5}


def _is_restoration(pre: State, post: State) -> bool:
    """rho: the transition *into* the restored phase."""
    return post.phase == RESTORED and pre.phase != RESTORED


def _is_advance(pre: State, post: State) -> bool:
    """sigma: any transition that moves the phase forward along
    DENIED -> R0 -> R1 -> R2 -> R3 -> RESTORED (review item M1)."""
    return (pre.phase in _ORDER and post.phase in _ORDER
            and pre.phase != RESTORED and _ORDER[post.phase] > _ORDER[pre.phase])


TransitionProperty = Callable[[State, State], bool]

#: P1-P8 and P10 are safety claims: each returns True when the transition is a
#: *violation*, so the witness count is the number of transitions returning True.
_COMMON: dict[str, tuple[str, TransitionProperty]] = {
    # P1 and P10 partition the illegitimate authority sources: a secret the
    # platform holds itself, and a holder coerced below the threshold. Neither
    # alone is "no restoration without legitimate authority" -- together they
    # are, because NONE never admits a restoration in either controller.
    "P1": ("No restoration from a secret held on the platform",
           lambda a, b: _is_restoration(a, b) and a.auth == ONBOARD),
    "P2": ("No token replay: a consumed authorisation cannot restore again",
           lambda a, b: _is_restoration(a, b) and a.used),
    "P3": ("No restoration while the adversary holds physical possession",
           lambda a, b: _is_restoration(a, b) and a.possessed),
    "P7": ("Denial is durable: no extraction during an active denial episode",
           lambda a, b: b.extracted and not a.extracted and a.denied),
    "P8": ("A tampered configuration is never restored to capability",
           lambda a, b: _is_restoration(a, b) and a.tampered),
    "P10": ("Coerced holders below the threshold cannot restore",
            lambda a, b: _is_restoration(a, b) and a.auth == COERCED),
    # P11 and P12 are the safety half of Requirement 3. Both are state
    # invariants rather than claims about a particular kind of step, and they
    # are written that way deliberately: as transition claims they would
    # constrain only the moves that *set* the field, leaving reachable states
    # that violate them through a later move which merely leaves it alone.
    "P11": ("Navigation capability survives every denial state",
            lambda a, b: b.denied and not b.nav_available),
    "P12": ("Command authority is never held while the adversary holds the "
            "platform",
            lambda a, b: b.possessed and b.command_owner != AUTONOMOUS),
}

#: The stage-wise P4-P6 (M1). The first version checked them on rho only, which
#: does not exclude a skip from R0 to R2 and so did not verify what Section
#: sec:staged claims of every stage.
_STAGEWISE = {
    "P4": ("Every stage advance happens inside the navigational safe-state envelope",
           lambda a, b: _is_advance(a, b) and not a.safe),
    "P5": ("Stages advance one at a time: R0-R3 is never skipped",
           lambda a, b: _is_advance(a, b)
           and _ORDER[b.phase] != _ORDER[a.phase] + 1),
    "P6": ("Nothing past R1 is entered without post-possession attestation",
           lambda a, b: _is_advance(a, b) and _ORDER[b.phase] >= _ORDER[R2]
           and not b.attested),
}
_FINAL_ONLY = {
    "P4": ("No restoration outside the navigational safe-state envelope",
           lambda a, b: _is_restoration(a, b) and not a.safe),
    "P5": ("Ordered staging R0-R3 is never bypassed",
           lambda a, b: _is_restoration(a, b) and a.phase != R3),
    "P6": ("Capability is restored only after post-possession attestation",
           lambda a, b: _is_restoration(a, b) and not a.attested),
}


def safety_properties(cfg: Settings = SETTINGS) -> dict[str, tuple[str, TransitionProperty]]:
    return {**_COMMON, **(_STAGEWISE if cfg.stagewise else _FINAL_ONLY)}


SAFETY_PROPERTIES = safety_properties()

P9_CLAIM = ("Restoration reachability: a legitimate path remains from every "
            "denial state, given eventual authority")

#: Moves only an adversary makes. A recovery path that has to traverse one of
#: these is not a recovery path, so P9's backward search excludes them. Without
#: this restriction the check is vacuous in the direction that matters: plain
#: ``AG(DENIED -> EF RESTORED)`` is satisfied by *any* route to RESTORED,
#: including one an adversary drives, and B0 -- whose on-board secret restores
#: the platform for whoever holds it -- would pass on the strength of the very
#: weakness the paper is about.
ADVERSARY_MOVES = frozenset({
    "seize", "modify", "extract", "coerce_holder", "read_onboard_secret",
    "steal_offline_credential", "steal_custody_proof", "present_stolen_offline",
    "extract_sk_tr", "decrypt_offline_opening",
})


def check_p9(rel: Relation) -> list[State]:
    """Backward reachability from a *legitimate* restoration.

    Returns the DENIED states from which no legitimate restoration is
    reachable: the witnesses to a violation of

        AG(phase = DENIED -> EF_L (phase = RESTORED /\\ auth in {LEGIT, OFFLINE}))

    where ``EF_L`` ranges over paths whose every transition lies outside
    :data:`ADVERSARY_MOVES`.
    """
    back: dict[State, list[State]] = {}
    for pre, label, post in rel.transitions:
        if label not in ADVERSARY_MOVES:
            back.setdefault(post, []).append(pre)

    # The goal is not "restored" but "restored by threshold-issued authority".
    seen = {post for pre, _, post in rel.transitions
            if _is_restoration(pre, post) and pre.auth in (LEGIT, OFFLINE)}
    queue = deque(seen)
    while queue:
        s = queue.popleft()
        for pre in back.get(s, ()):
            if pre not in seen:
                seen.add(pre)
                queue.append(pre)

    return sorted((s for s in rel.states if s.phase == DENIED and s not in seen),
                  key=repr)


@dataclass(frozen=True)
class VariantResult:
    variant: str
    n_states: int
    n_transitions: int
    witnesses: dict[str, int]
    p9_unrecoverable: tuple[State, ...]

    @property
    def satisfied(self) -> dict[str, bool]:
        return {k: v == 0 for k, v in self.witnesses.items()}

    @property
    def n_satisfied(self) -> int:
        return sum(self.satisfied.values())


def _check(v: Variant, cfg: Settings, mut: frozenset = frozenset()) -> VariantResult:
    rel = explore(v, cfg=cfg, mut=mut)
    witnesses = {
        name: sum(1 for pre, _, post in rel.transitions if pred(pre, post))
        for name, (_, pred) in safety_properties(cfg).items()
    }
    bad = check_p9(rel)
    witnesses["P9"] = len(bad)
    ordered = {f"P{i}": witnesses[f"P{i}"] for i in range(1, 13)}
    return VariantResult(
        variant=v.name,
        n_states=len(rel.states),
        n_transitions=len(rel.transitions),
        witnesses=ordered,
        p9_unrecoverable=tuple(bad),
    )


def verify(variant_name: str, cfg: Settings | None = None) -> VariantResult:
    """Run the exhaustive check for one controller variant."""
    return _check(VARIANTS[variant_name], cfg or SETTINGS)


FLAGS = ("erase_keys_on_denial", "onboard_recovery_secret", "monotonic_counter",
         "staged", "gate_custody", "gate_safe", "gate_attested", "gate_unused",
         "require_threshold", "nav_kernel_outside_denial")


def ablate(variant_name: str = "H1", cfg: Settings | None = None) -> dict[str, tuple[str, ...]]:
    """Which of the variant's flags actually discharges which property?

    A verdict of 12/12 says the design as a whole is sound; it does not say
    which part of it does the work, and the answer is not always the design
    feature the prose would suggest. This flips each flag in turn and reports
    the properties that break, so the mechanism column of Table `tab:mc` can be
    checked rather than asserted.

    Returns ``{flag: (properties that break when it is flipped)}``.
    """
    cfg = cfg or SETTINGS
    base = VARIANTS[variant_name]
    baseline = _check(base, cfg).satisfied
    out: dict[str, tuple[str, ...]] = {}
    for flag in FLAGS:
        probe = Variant(**{**base.__dict__, flag: not getattr(base, flag)})
        got = _check(probe, cfg).witnesses
        out[flag] = tuple(p for p in PROPERTY_CLAIMS
                          if baseline.get(p, False) and got[p] != 0)
    return out


#: One mutant per mechanism (review item M1, "mutation testing per property").
#: Each is H1 with one flag flipped or one rule of the relation switched off.
MUTANTS: dict[str, tuple[str, dict, frozenset]] = {
    "onboard_no_threshold": ("on-board secret and no threshold enforcement",
                             dict(onboard_recovery_secret=True, require_threshold=False), frozenset()),
    "no_counter": ("no monotonic counter, no unused-token check",
                   dict(monotonic_counter=False, gate_unused=False), frozenset()),
    "no_custody": ("no custody conjunct", dict(gate_custody=False), frozenset()),
    "no_safe": ("no per-stage safe-state gate", dict(gate_safe=False), frozenset()),
    "single_step": ("single-step restoration", dict(staged=False), frozenset()),
    "no_r1_check": ("R1 does not compare the measured state", {}, frozenset({"no_r1_check"})),
    "no_erase": ("keys not erased at denial", dict(erase_keys_on_denial=False), frozenset()),
    "residue": ("keys erased but a residue remains (no X4)", {}, frozenset({"residue"})),
    "no_threshold": ("below-threshold authority accepted", dict(require_threshold=False), frozenset()),
    "kernel_inside": ("navigation kernel inside the denial boundary",
                      dict(nav_kernel_outside_denial=False), frozenset()),
    "no_cmd_revert": ("command not reverted to the kernel on seizure", {}, frozenset({"no_cmd_revert"})),
    "no_reissue": ("stale legitimate authority cannot be re-issued", {}, frozenset({"no_reissue"})),
}


def mutation_test(cfg: Settings | None = None) -> dict[str, tuple[str, ...]]:
    """``{mutant: properties it breaks}``. Every property should be broken by at
    least one mutant, or its check is vacuous."""
    cfg = cfg or SETTINGS
    base = VARIANTS["H1"]
    out = {}
    for name, (_, flags, mut) in MUTANTS.items():
        v = Variant(**{**base.__dict__, **flags, "name": "H1-" + name})
        w = _check(v, cfg, mut).witnesses
        out[name] = tuple(p for p, n in w.items() if n)
    return out


def counterexample(variant_name: str, prop: str, limit: int = 1,
                   cfg: Settings | None = None) -> list[tuple[State, str, State]]:
    """Return up to ``limit`` violating transitions, for inspection.

    A witness count alone is not evidence; a reviewer should be able to read the
    concrete transition that breaks the claim.
    """
    cfg = cfg or SETTINGS
    if prop == "P9":
        raise ValueError("P9 witnesses are states, not transitions; "
                         "use VariantResult.p9_unrecoverable")
    _, pred = safety_properties(cfg)[prop]
    rel = explore(VARIANTS[variant_name], cfg=cfg)
    out = [e for e in rel.transitions if pred(e[0], e[2])]
    return out[:limit]


PROPERTY_CLAIMS = {name: claim for name, (claim, _) in SAFETY_PROPERTIES.items()}
PROPERTY_CLAIMS["P9"] = P9_CLAIM
PROPERTY_CLAIMS = {f"P{i}": PROPERTY_CLAIMS[f"P{i}"] for i in range(1, 13)}
