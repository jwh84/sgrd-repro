"""The restoration controller re-encoded in the NuSMV input language, for an
independent cross-check of :mod:`sgrd.modelcheck` (review item M1).

The transition rules are written out a second time as an SMV ``TRANS`` with the
move chosen by an input variable; transition properties are checked as
``INVARSPEC`` over previous-state variables ``p_*``, and P9 as the CTL formula
``AG(phase = DENIED -> EX E[!adv_step U restored_legit])``. NuSMV checks the
model symbolically (BDDs), the Python module by explicit enumeration, so
agreement is evidence that neither verdict is an artefact of one encoding.

Usage::

    python -m sgrd.smv C3 H1 > h1.smv        # write one model
    NuSMV -dcx h1.smv                         # check it (NuSMV 2.7.1)

Settings: C0 = first published version, C3 = thesis main result,
C4/C5 = one/both offline artefacts stolen, C6 = one stolen + sk_tr extractable.
"""
import re
import sys

VARIANTS = {
    #        erase onboard counter staged custody safe attest unused thr kernel offline
    'B0': (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    'K1': (1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0),
    'K2': (1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0),
    'K3': (1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
    'B1': (1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1),
    'B2': (1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1),
    'B3': (1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1),
    'H1': (1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1),
}
FLAGS = ['ERASE', 'ONBOARD', 'COUNTER', 'STAGED', 'G_CUST', 'G_SAFE', 'G_ATT', 'G_UNUSED', 'THR', 'KERNEL_OUT',
         'OFFPATH']
CONFIGS = {
    # stagewise false_trigger offline theft(0 none,1 one,2 both) sktr
    'C0': (0, 0, 0, 0, 0),
    'C3': (1, 1, 1, 0, 0),
    'C4': (1, 1, 1, 1, 0),
    'C5': (1, 1, 1, 2, 0),
    'C6': (1, 1, 1, 1, 1),
}

VARS = [('phase', '{NORMAL, DENIED, R0, R1, R2, R3, RESTORED}', 'NORMAL'),
        ('possessed', 'boolean', 'FALSE'), ('tampered', 'boolean', 'FALSE'), ('attested', 'boolean', 'TRUE'),
        ('keys', 'boolean', 'TRUE'), ('extracted', 'boolean', 'FALSE'),
        ('auth', '{none, legit, coerced, onboard, offline, offadv}', 'none'),
        ('fresh', 'boolean', 'FALSE'), ('safe', 'boolean', 'TRUE'), ('used', 'boolean', 'FALSE'),
        ('denied', 'boolean', 'FALSE'), ('nav', 'boolean', 'TRUE'), ('cmd', '{roc, boarding, auto}', 'roc'),
        ('sensor', 'boolean', 'TRUE'), ('off_stolen', 'boolean', 'FALSE'), ('cust_stolen', 'boolean', 'FALSE'),
        ('sktr', 'boolean', 'FALSE'), ('off_spent', 'boolean', 'FALSE')]
HIST = ['phase', 'possessed', 'tampered', 'attested', 'extracted', 'auth', 'safe', 'used', 'denied']
STAGES = '{R0, R1, R2, R3}'

ADV = {'seize', 'modify', 'extract', 'coerce', 'read_onboard', 'steal_off', 'steal_cust', 'present_stolen',
       'extract_sktr', 'decrypt_off'}


def build(cfg, var):
    sw, ft, off, theft, sk = CONFIGS[cfg]
    f = dict(zip(FLAGS, VARIANTS[var]))
    B = lambda x: 'TRUE' if x else 'FALSE'
    offp = bool(off and f['OFFPATH'])
    gate = '((!G_SAFE | safe) & (!G_CUST | !possessed | auth = offadv))'
    auth_ok = '(THR & auth in {legit, offline, offadv} | !THR & auth != none)'
    # (이름, 가드, {변수: 다음 값})
    A = []
    A.append(('seize', '!possessed', dict(possessed='TRUE', attested='FALSE', sensor='FALSE', cmd='auto',
                                           phase='(phase in %s ? DENIED : phase)' % STAGES)))
    A.append(('regain', 'possessed', dict(possessed='FALSE')))
    A.append(('modify', 'possessed & !tampered', dict(tampered='TRUE', attested='FALSE')))
    A.append(('extract', 'possessed & keys & !extracted', dict(extracted='TRUE')))
    if offp and sk:
        A.append(('extract_sktr', 'possessed & !sktr', dict(sktr='TRUE')))
    if offp:
        A.append(('decrypt_off', 'possessed & sktr & off_stolen & !extracted', dict(extracted='TRUE')))
    A.append(('flip_safe', 'TRUE', dict(safe='!safe')))
    A.append(('stale', 'fresh', dict(fresh='FALSE')))
    A.append(('issue', 'denied & !possessed & !(auth = legit & fresh)', dict(auth='legit', fresh='TRUE')))
    A.append(('coerce', 'denied & auth = none', dict(auth='coerced', fresh='TRUE')))
    A.append(('remediate', 'tampered & !possessed', dict(tampered='FALSE')))
    A.append(('read_onboard', 'ONBOARD & denied & auth = none', dict(auth='onboard', fresh='TRUE')))
    A.append(('board_cmd', 'denied & !possessed & nav & cmd = auto', dict(cmd='boarding')))
    if offp:
        A.append(('present_off', 'denied & !possessed & !off_spent & !off_stolen & auth != offline',
                  dict(auth='offline', fresh='TRUE')))
        if theft >= 1:
            one = '' if theft == 2 else ' & !cust_stolen'
            A.append(('steal_off', '!off_stolen' + one, dict(off_stolen='TRUE')))
            one = '' if theft == 2 else ' & !off_stolen'
            A.append(('steal_cust', '!cust_stolen' + one, dict(cust_stolen='TRUE')))
        A.append(('present_stolen', 'possessed & denied & off_stolen & cust_stolen & !off_spent & auth = none',
                  dict(auth='offadv', fresh='TRUE')))
    deny = dict(phase='DENIED', denied='TRUE', auth='none', fresh='FALSE', keys='!ERASE',
                used='(COUNTER ? FALSE : used)', nav='KERNEL_OUT', cmd='auto')
    A.append(('deny', 'phase = NORMAL & possessed & !denied', deny))
    if ft:
        A.append(('spurious', 'phase = NORMAL & !possessed & !denied', deny))
    restore_common = dict(phase='RESTORED', keys='TRUE', used='TRUE', denied='FALSE', off_spent='FALSE',
                          off_stolen='FALSE')
    # 단계 제어기
    A.append(('r0_admit', 'STAGED & phase = DENIED & %s & fresh & %s & (!G_UNUSED | !used)' % (auth_ok, gate),
              dict(phase='R0', off_spent='(auth in {offline, offadv} ? TRUE : off_spent)')))
    A.append(('r0_r1', 'STAGED & phase = R0 & %s' % gate, dict(phase='R1')))
    A.append(('r1_bad', 'STAGED & phase = R1 & tampered', dict(phase='DENIED')))
    A.append(('r1_ok', 'STAGED & phase = R1 & !tampered & %s' % gate, dict(phase='R2', attested='TRUE')))
    A.append(('r2_r3', 'STAGED & phase = R2 & %s & (!G_ATT | attested)' % gate,
              dict(phase='R3', sensor='TRUE', cmd='roc')))
    A.append(('r3_restore', 'STAGED & phase = R3 & %s & (!G_ATT | attested) & sensor' % gate,
              dict(restore_common, auth='none', fresh='FALSE')))
    A.append(('abort', 'STAGED & phase in %s & G_SAFE & !safe' % STAGES, dict(phase='DENIED')))
    # 한 번에 복원
    A.append(('restore1', '!STAGED & phase = DENIED & %s & %s & (!G_UNUSED | !used)' % (auth_ok, gate),
              dict(restore_common, sensor='TRUE', cmd='roc')))
    A.append(('resume', 'phase = RESTORED', dict(phase='NORMAL')))

    L = ['MODULE main', 'DEFINE']
    for k in FLAGS:
        L.append('  %s := %s;' % (k, B(f[k])))
    L.append('VAR')
    for n, t, _ in VARS:
        L.append('  %s : %s;' % (n, t))
    for n in HIST:
        t = dict((a, b) for a, b, _ in VARS)[n]
        L.append('  p_%s : %s;' % (n, t))
    L.append('  adv_step : boolean;')
    L.append('IVAR')
    L.append('  act : {%s};' % ', '.join(a for a, _, _ in A))
    L.append('ASSIGN')
    for n, _, i in VARS:
        L.append('  init(%s) := %s;' % (n, i))
    for n in HIST:
        L.append('  init(p_%s) := %s;' % (n, dict((a, c) for a, _, c in VARS)[n]))
        L.append('  next(p_%s) := %s;' % (n, n))
    L.append('  init(adv_step) := FALSE;')
    L.append('TRANS')
    parts = []
    for a, g, upd in A:
        nx = ' & '.join('next(%s) = (%s)' % (n, upd.get(n, n)) for n, _, _ in VARS)
        parts.append('  (act = %s & (%s) & %s & next(adv_step) = %s)' % (a, g, nx, B(a in ADV)))
    L.append('\n  | '.join(parts) + ';')
    # 속성
    rho = '(p_phase != RESTORED & phase = RESTORED)'
    L.append('DEFINE')
    L.append('  ordp := case p_phase = DENIED : 0; p_phase = R0 : 1; p_phase = R1 : 2; p_phase = R2 : 3; '
             'p_phase = R3 : 4; p_phase = RESTORED : 5; TRUE : -1; esac;')
    L.append('  ordc := case phase = DENIED : 0; phase = R0 : 1; phase = R1 : 2; phase = R2 : 3; '
             'phase = R3 : 4; phase = RESTORED : 5; TRUE : -1; esac;')
    L.append('  adv := ordp >= 0 & ordc > ordp & p_phase != RESTORED;')
    L.append('  rho := %s;' % rho)
    L.append('  restored_legit := rho & p_auth in {legit, offline};')
    P = {
        'P1': 'rho & p_auth = onboard', 'P2': 'rho & p_used', 'P3': 'rho & p_possessed',
        'P7': 'extracted & !p_extracted & p_denied', 'P8': 'rho & p_tampered', 'P10': 'rho & p_auth = coerced',
        'P11': 'denied & !nav', 'P12': 'possessed & cmd != auto',
    }
    if sw:
        P['P4'] = 'adv & !p_safe'
        P['P5'] = 'adv & ordc != ordp + 1'
        P['P6'] = 'adv & ordc >= 3 & !attested'
    else:
        P['P4'] = 'rho & !p_safe'
        P['P5'] = 'rho & p_phase != R3'
        P['P6'] = 'rho & !p_attested'
    for k in ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P10', 'P11', 'P12']:
        L.append('INVARSPEC NAME %s := !(%s)' % (k, P[k]))
    L.append('CTLSPEC NAME P9 := AG (phase = DENIED -> EX E [ !adv_step U restored_legit ])')
    return '\n'.join(L) + '\n'


def verdicts(output: str, model: str) -> dict[str, bool]:
    """Parse NuSMV output for a model written by :func:`build`."""
    norm = lambda x: re.sub(r'[\s()]', '', x)
    names = {norm(e): n for n, e in re.findall(r'INVARSPEC NAME (\w+) := !\((.*)\)\n', model)}
    out = {names[norm(e)]: v == 'true' for e, v in re.findall(r'-- invariant !\((.*)\)\s+is (true|false)', output)}
    m = re.search(r'-- specification AG .* is (true|false)', output)
    out['P9'] = bool(m) and m.group(1) == 'true'
    return out


if __name__ == '__main__':
    sys.stdout.write(build(sys.argv[1], sys.argv[2]))
