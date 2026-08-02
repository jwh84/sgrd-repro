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


INITIAL = State(
    phase=NORMAL, possessed=False, tampered=False, attested=True, keys=True,
    extracted=False, auth=NONE, fresh=False, safe=True, used=False, denied=False,
    nav_available=True, command_owner=ROC, sensor_trusted=True,
)


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
    nav_kernel_outside_denial=True,
)

B2 = Variant(  # threshold authority, single-step restoration
    name="B2", erase_keys_on_denial=True, onboard_recovery_secret=False,
    monotonic_counter=True, staged=False, gate_custody=True, gate_safe=True,
    gate_attested=True, gate_unused=True, require_threshold=True,
    nav_kernel_outside_denial=True,
)

B3 = Variant(  # crypto-erasure and threshold authority, no safe-state gate
    name="B3", erase_keys_on_denial=True, onboard_recovery_secret=False,
    monotonic_counter=True, staged=True, gate_custody=True, gate_safe=False,
    gate_attested=True, gate_unused=True, require_threshold=True,
    nav_kernel_outside_denial=True,
)

VARIANTS = {"B0": B0, "B1": B1, "B2": B2, "B3": B3, "H1": H1}


# --- Transition relation -----------------------------------------------------

def successors(s: State, v: Variant) -> Iterator[tuple[str, State]]:
    """All transitions enabled in ``s``, as ``(label, successor)`` pairs.

    Environment moves come first, then the denial actuator, then the controller.
    Nothing here is scheduled: every enabled move is explored, which is what
    makes the check exhaustive rather than a simulation.
    """
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
                      command_owner=AUTONOMOUS)
        if s.phase in STAGES:
            nxt = replace(nxt, phase=DENIED)
        yield "seize", nxt
    else:
        yield "regain_custody", replace(s, possessed=False)

        if not s.tampered:
            # E3 modification in captivity (hazard 4 of Section sec:staged).
            yield "modify", replace(s, tampered=True, attested=False)

        if s.keys and not s.extracted:
            # E4 extraction. Requires the hardware *and* usable key material;
            # this is the transition H1 removes by erasing K_w at denial.
            yield "extract", replace(s, extracted=True)

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
    # modelling artefact and it manufactures a P9 violation -- twenty DENIED
    # states with legitimate-but-stale authority and no way to refresh it, from
    # which restoration is genuinely unreachable in the model and perfectly
    # reachable in the design. Exhaustive checking found this; a simulation
    # would not have.
    if s.denied and not s.possessed and not (s.auth == LEGIT and s.fresh):
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

    # --- Denial actuator ----------------------------------------------------

    if s.phase == NORMAL and s.possessed and not s.denied:
        nxt = replace(
            s, phase=DENIED, denied=True, auth=NONE, fresh=False,
            # H1 erases K_w and the helper datum; B0 retains sealed material.
            keys=not v.erase_keys_on_denial,
            # A monotonic counter opens a fresh token space per episode. Without
            # one, a previously consumed authorisation is still current.
            used=False if v.monotonic_counter else s.used,
            # The boundary of Section `sec:kernel`. Where the kernel lies
            # outside it, denial leaves the vessel navigable and the kernel
            # holds a minimum-risk condition without being commanded. Where it
            # does not -- B0's blunt lockout -- denial takes navigation with it,
            # which is what P11 detects.
            nav_available=v.nav_kernel_outside_denial,
            command_owner=AUTONOMOUS,
        )
        yield "deny", nxt

    # --- Restoration controller --------------------------------------------

    if v.staged:
        yield from _staged_controller(s, v)
    else:
        yield from _single_step_controller(s, v)

    # Normal operation resumes after a restoration, so a later seizure can fire
    # denial again. Without this the model could not express repeated episodes,
    # and P9 would be checked over a single episode only.
    if s.phase == RESTORED:
        yield "resume_normal_operation", replace(s, phase=NORMAL)


def _gate(s: State, v: Variant) -> bool:
    """The conjuncts common to every stage of the hardened controller."""
    return ((not v.gate_safe or s.safe)
            and (not v.gate_custody or not s.possessed))


def _staged_controller(s: State, v: Variant) -> Iterator[tuple[str, State]]:
    if s.phase == DENIED:
        # R0 admission: Auth AND Safe AND NOT Possessed AND NOT Used.
        authority_ok = s.auth == LEGIT if v.require_threshold else s.auth != NONE
        if (authority_ok and s.fresh and _gate(s, v)
                and (not v.gate_unused or not s.used)):
            yield "R0_admit", replace(s, phase=R0)

    elif s.phase == R0:
        if _gate(s, v):
            yield "R0_to_R1", replace(s, phase=R1)

    elif s.phase == R1:
        # Re-attestation. Nothing from A_cpi executes in this stage.
        if s.tampered:
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
                auth=NONE, fresh=False,
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
        authority_ok = s.auth == LEGIT if v.require_threshold else s.auth != NONE
        if (authority_ok and _gate(s, v)
                and (not v.gate_unused or not s.used)):
            yield "restore", replace(
                s, phase=RESTORED, keys=True, used=True, denied=False,
                sensor_trusted=True, command_owner=ROC,
            )


# --- Reachability ------------------------------------------------------------

class Relation(NamedTuple):
    states: set[State]
    transitions: list[tuple[State, str, State]]


def explore(v: Variant, initial: State = INITIAL) -> Relation:
    """BFS the full reachable state set and collect every reachable transition."""
    seen = {initial}
    queue = deque([initial])
    edges: list[tuple[State, str, State]] = []
    while queue:
        s = queue.popleft()
        for label, t in successors(s, v):
            edges.append((s, label, t))
            if t not in seen:
                seen.add(t)
                queue.append(t)
    return Relation(seen, edges)


# --- Properties (Appendix app:props) ----------------------------------------

def _is_restoration(pre: State, post: State) -> bool:
    """rho: the transition *into* the restored phase."""
    return post.phase == RESTORED and pre.phase != RESTORED


TransitionProperty = Callable[[State, State], bool]

#: P1-P8 and P10 are safety claims: each returns True when the transition is a
#: *violation*, so the witness count is the number of transitions returning True.
SAFETY_PROPERTIES: dict[str, tuple[str, TransitionProperty]] = {
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
    "P4": ("No restoration outside the navigational safe-state envelope",
           lambda a, b: _is_restoration(a, b) and not a.safe),
    "P5": ("Ordered staging R0-R3 is never bypassed",
           lambda a, b: _is_restoration(a, b) and a.phase != R3),
    "P6": ("Capability is restored only after post-possession attestation",
           lambda a, b: _is_restoration(a, b) and not a.attested),
    "P7": ("Denial is durable: no extraction during an active denial episode",
           lambda a, b: b.extracted and not a.extracted and a.denied),
    "P8": ("A tampered configuration is never restored to capability",
           lambda a, b: _is_restoration(a, b) and a.tampered),
    "P10": ("Coerced holders below the threshold cannot restore",
            lambda a, b: _is_restoration(a, b) and a.auth == COERCED),
    # P11 and P12 are the safety half of Requirement 3, which until now was
    # argued in prose and checked only through the safe-state Boolean. Both are
    # state invariants rather than claims about a particular kind of step, and
    # they are written that way deliberately: as transition claims they would
    # constrain only the moves that *set* the field, leaving reachable states
    # that violate them through a later move which merely leaves it alone.
    "P11": ("Navigation capability survives every denial state",
            lambda a, b: b.denied and not b.nav_available),
    "P12": ("Command authority is never held while the adversary holds the "
            "platform",
            lambda a, b: b.possessed and b.command_owner != AUTONOMOUS),
}

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
})


def check_p9(rel: Relation) -> list[State]:
    """Backward reachability from a *legitimate* restoration.

    Returns the DENIED states from which no legitimate restoration is
    reachable: the witnesses to a violation of

        AG(phase = DENIED -> EF_L (phase = RESTORED /\\ auth = LEGIT))

    where ``EF_L`` ranges over paths whose every transition lies outside
    :data:`ADVERSARY_MOVES`.
    """
    back: dict[State, list[State]] = {}
    for pre, label, post in rel.transitions:
        if label not in ADVERSARY_MOVES:
            back.setdefault(post, []).append(pre)

    # The goal is not "restored" but "restored by threshold-issued authority".
    seen = {post for pre, _, post in rel.transitions
            if _is_restoration(pre, post) and pre.auth == LEGIT}
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


def verify(variant_name: str) -> VariantResult:
    """Run the exhaustive check for one controller variant."""
    v = VARIANTS[variant_name]
    rel = explore(v)

    witnesses = {
        name: sum(1 for pre, _, post in rel.transitions if pred(pre, post))
        for name, (_, pred) in SAFETY_PROPERTIES.items()
    }
    bad = check_p9(rel)
    witnesses["P9"] = len(bad)

    ordered = {f"P{i}": witnesses[f"P{i}"] for i in range(1, 13)}
    return VariantResult(
        variant=variant_name,
        n_states=len(rel.states),
        n_transitions=len(rel.transitions),
        witnesses=ordered,
        p9_unrecoverable=tuple(bad),
    )


FLAGS = ("erase_keys_on_denial", "onboard_recovery_secret", "monotonic_counter",
         "staged", "gate_custody", "gate_safe", "gate_attested", "gate_unused",
         "require_threshold", "nav_kernel_outside_denial")


def ablate(variant_name: str = "H1") -> dict[str, tuple[int, ...]]:
    """Which of the variant's flags actually discharges which property?

    A verdict of 10/10 says the design as a whole is sound; it does not say
    which part of it does the work, and the answer is not always the design
    feature the prose would suggest. This flips each flag in turn and reports
    the properties that break, so the mechanism column of Table `tab:mc` can be
    checked rather than asserted.

    Returns ``{flag: (properties that break when it is flipped)}``.
    """
    base = VARIANTS[variant_name]
    baseline = verify(variant_name).satisfied
    out: dict[str, tuple[int, ...]] = {}
    for flag in FLAGS:
        probe = Variant(**{**base.__dict__, flag: not getattr(base, flag)})
        rel = explore(probe)
        w = {n: sum(1 for a, _, b in rel.transitions if pred(a, b))
             for n, (_, pred) in SAFETY_PROPERTIES.items()}
        w["P9"] = len(check_p9(rel))
        out[flag] = tuple(p for p in PROPERTY_CLAIMS
                          if baseline.get(p, False) and w[p] != 0)
    return out


def counterexample(variant_name: str, prop: str, limit: int = 1) -> list[tuple[State, str, State]]:
    """Return up to ``limit`` violating transitions, for inspection.

    A witness count alone is not evidence; a reviewer should be able to read the
    concrete transition that breaks the claim.
    """
    if prop == "P9":
        raise ValueError("P9 witnesses are states, not transitions; "
                         "use VariantResult.p9_unrecoverable")
    _, pred = SAFETY_PROPERTIES[prop]
    rel = explore(VARIANTS[variant_name])
    out = [e for e in rel.transitions if pred(e[0], e[2])]
    return out[:limit]


PROPERTY_CLAIMS = {name: claim for name, (claim, _) in SAFETY_PROPERTIES.items()}
PROPERTY_CLAIMS["P9"] = P9_CLAIM
