# -*- coding: utf-8 -*-
"""모델–구현 일치 시험(conformance testing).

Conformance test between sgrd.modelcheck and sgrd.refimpl.protocol (review item M1).

형식 모델(modelcheck, H1, 단계별 속성 + 오발동, 오프라인 경로 없음)의 무작위 사건 순서를 참조 구현(sgrd/refimpl/protocol.py)에
그대로 넣고, 제어기의 판단(받아들임/거부)과 그 뒤의 상태(단계, 무력화 여부, 점유, 작업 키 존재)가 모델과 같은지 본다.

대응 규칙(구현이 모델보다 구체적이어서 생기는 차이를 '구현이 더 엄격한 쪽'으로 대응시킴):
  R1 구현은 중단(나포·불안전·증명 불일치)하면 회차 자료(논스, 임시 키, 토큰)를 지운다 → 모델에서는 인가 만료(fresh=False).
  R2 토큰은 요청 시점의 안전 증거와 측정값에 묶인다 → 받아들이기 전에 안전 상황이나 구성이 바뀌면 인가 만료.
  R3 단계 진행 중 안전 상황이 바뀌면 구현의 안전 증거를 새 값으로 바꾼다(단계마다 다시 평가하는 것에 해당).
  R4 구현은 단계 중 나포를 다음 단계 시도 때 감지한다 → 나포 직후 감지 함수를 바로 불러 모델과 시점을 맞춘다.
  R5 인가 발급(정당·협박)은 DENIED 단계에서만 일으킨다(단계 진행 중 새 요청은 구현에서 진행 중 회차를 덮어씀).
"""
import os, random, sys
from dataclasses import replace as dreplace
from .. import modelcheck as M
from . import protocol as pr

#: The reference implementation has no offline path, so it is compared with the
#: model's online controller: stage-wise properties and false triggers, no
#: Algorithm 2.
CFG = M.Settings(offline=False)
H1 = M.VARIANTS['H1']
TAMPERED = {"autonomy-stack": pr._h(b"art", b"cpi-stack-EVIL")}

ENV = ['seize', 'regain', 'modify', 'remediate', 'safe_flip', 'issue', 'coerce', 'stale', 'deny', 'spurious']
CTRL = ['admit', 'to_r1', 'attest', 'revalidate', 'restore', 'resume', 'gate_check']
MODEL_LABEL = {'admit': ['R0_admit'], 'to_r1': ['R0_to_R1'],
               'attest': ['R1_attest_ok', 'R1_attestation_mismatch'], 'revalidate': ['R2_to_R3'],
               'restore': ['R3_restore'], 'resume': ['resume_normal_operation'], 'gate_check': ['abort_unsafe']}


ENV_LABEL = {'seize': ['seize'], 'regain': ['regain_custody'], 'modify': ['modify'],
             'remediate': ['off_board_remediation'], 'safe_flip': ['safe_lost', 'safe_regained'],
             'issue': ['issue_legitimate_authority'], 'coerce': ['coerce_holder'], 'stale': ['authorisation_stale'],
             'deny': ['deny'], 'spurious': ['spurious_deny']}


def succ(s):
    return {lab: t for lab, t in M.successors(s, H1, CFG)}


class Harness:
    def __init__(self, rng):
        self.rng = rng
        self.sc = pr.build_scenario(t=3)
        self.p = self.sc.platform
        self.s = M.INITIAL
        self.bundle = None          # (token, custody, reencrypted)
        self.safe = True
        self.log = []

    # ---- 구현 쪽 보조 ----
    def evidence(self):
        return pr.SafeEvidence("open-water", self.safe, self.safe)

    def make_bundle(self, signers):
        p, q = self.p, self.sc.quorum
        req = p.request_recovery(self.evidence())
        cust = None
        if not p.possessed:      # 발급자는 관리권이 돌아왔을 때만 증명한다
            cust = q.issuer.issue(vessel_id=p.vessel_id, counter=p.counter, nonce=p.nonce,
                                  boarding_ref=b"conf", issued_at=p.clock())
        tr = dreplace(req.transcript, h_cust=cust.digest() if cust else b"")
        tok = pr.issue_token(q, tr, signers)
        try:
            re = pr.reencrypt(q, req.capsule, p.pk_eph, signers)
        except Exception:
            re = None
        return (tok, cust, re)

    def stale_model(self):
        if self.s.fresh:
            self.s = M.replace(self.s, fresh=False)

    def proj_impl(self):
        p = self.p
        return (p.phase, p.denied, p.possessed, p.working_keys is not None)

    def proj_model(self):
        s = self.s
        return (s.phase, s.denied, s.possessed, s.keys)

    # ---- 환경 사건(모델에서 가능한 때만 둘 다에 적용) ----
    def env(self, e):
        s, p = self.s, self.p
        nx = succ(s)
        if e == 'seize' and 'seize' in nx:
            self.s = nx['seize']; p.possessed = True
            if p.phase in pr.STAGES:          # R4
                try:
                    p._gate()
                except pr.Refused:
                    pass
                self.stale_model()            # R1
            return True
        if e == 'regain' and 'regain_custody' in nx:
            self.s = nx['regain_custody']; p.possessed = False; return True
        if e == 'modify' and 'modify' in nx:
            self.s = nx['modify']; p.manifest = pr.Manifest({**p.baseline.entries, **TAMPERED})
            if self.bundle is not None and p.phase == pr.DENIED:
                self.bundle = None; self.stale_model()      # R2
            return True
        if e == 'remediate' and 'off_board_remediation' in nx:
            self.s = nx['off_board_remediation']; p.manifest = p.baseline
            if self.bundle is not None and p.phase == pr.DENIED:
                self.bundle = None; self.stale_model()      # R2
            return True
        if e == 'safe_flip':
            lab = 'safe_lost' if s.safe else 'safe_regained'
            self.s = nx[lab]; self.safe = not self.safe
            if p.phase in pr.STAGES and p.pending_safe is not None:
                p.pending_safe = self.evidence()            # R3
            elif self.bundle is not None:
                self.bundle = None; self.stale_model()      # R2
            return True
        if e == 'stale' and 'authorisation_stale' in nx:
            self.s = nx['authorisation_stale']; self.bundle = None; return True
        if e == 'issue' and 'issue_legitimate_authority' in nx and p.phase == pr.DENIED:   # R5
            self.bundle = self.make_bundle([h.index for h in self.sc.quorum.holders[:3]])
            self.s = nx['issue_legitimate_authority']; return True
        if e == 'coerce' and 'coerce_holder' in nx and p.phase == pr.DENIED:
            self.bundle = self.make_bundle([h.index for h in self.sc.quorum.holders[:2]])   # t-1 명
            self.s = nx['coerce_holder']; return True
        if e == 'deny' and 'deny' in nx:
            self.s = nx['deny']; p.deny(); self.bundle = None; return True
        if e == 'spurious' and 'spurious_deny' in nx:
            self.s = nx['spurious_deny']; p.deny(); self.bundle = None; return True
        return False

    # ---- 제어기 시도(구현에는 늘 시도, 모델은 해당 전이가 있는지) ----
    def ctrl(self, c):
        p = self.p
        nx = succ(self.s)
        labs = [l for l in MODEL_LABEL[c] if l in nx]
        model_ok = bool(labs)
        reason = None
        try:
            if c == 'admit':
                if self.bundle is None:
                    raise pr.Refused('no_bundle')
                tok, cust, re = self.bundle
                if re is None:
                    raise pr.Refused('reencrypt_failed')
                p.admit(tok, cust, re)
            elif c == 'to_r1':
                p.to_r1()
            elif c == 'attest':
                p.attest()
            elif c == 'revalidate':
                p.revalidate_sensing()
            elif c == 'restore':
                p.restore()
            elif c == 'resume':
                p.resume_normal_operation()
            elif c == 'gate_check':
                if p.phase not in pr.STAGES:
                    raise pr.Refused('not_in_stage')
                p._gate()
                raise pr.Refused('gate_passed')   # 중단이 일어나지 않음
            impl_ok = True
        except pr.Refused as r:
            impl_ok = False
            reason = r.reason if hasattr(r, 'reason') else str(r)
        # 결과 대조
        if c == 'attest':
            # 모델: 정상(R2) 또는 불일치(DENIED). 구현: 성공=R2, attestation_mismatch=DENIED
            if 'R1_attestation_mismatch' in nx:
                # 둘 다 거부하고 DENIED 로 간다. 변조와 불안전(또는 나포)이 겹치면 모델은 변조를,
                # 구현은 단계 게이트를 먼저 보므로 거부 사유만 다를 수 있다(판단은 같음).
                agree = (not impl_ok and reason in ('attestation_mismatch', 'aborted_unsafe',
                                                     'aborted_possessed'))
                self.s = nx['R1_attestation_mismatch']; self.stale_model(); self.bundle = None
                return agree, model_ok, impl_ok, reason
            if 'R1_attest_ok' in nx:
                agree = impl_ok
                if impl_ok:
                    self.s = nx['R1_attest_ok']
                return agree, model_ok, impl_ok, reason
        if c == 'gate_check':
            impl_abort = reason in ('aborted_unsafe', 'aborted_possessed')
            agree = (impl_abort == model_ok)
            if model_ok and impl_abort:
                self.s = nx['abort_unsafe']; self.stale_model(); self.bundle = None
            return agree, model_ok, impl_abort, reason
        if reason in ('aborted_unsafe', 'aborted_possessed'):
            # 구현이 단계 시도 중 중단: 모델의 중단 전이와 맞아야 한다
            agree = (not model_ok) and ('abort_unsafe' in nx)
            if 'abort_unsafe' in nx:
                self.s = nx['abort_unsafe']
            self.stale_model(); self.bundle = None
            return agree, model_ok, impl_ok, reason
        agree = (model_ok == impl_ok)
        if impl_ok and model_ok:
            self.s = nx[labs[0]]
            if c == 'restore':
                self.bundle = None
        return agree, model_ok, impl_ok, reason


def run(walks=300, steps=60, seed=20261114):
    rng = random.Random(seed)
    stats = dict(ctrl=0, ctrl_ok_both=0, ctrl_refused_both=0, env=0, mismatch=0, proj_mismatch=0)
    by_ctrl = {c: [0, 0, 0] for c in CTRL}   # 시도, 모델 허용, 불일치
    reasons = {}
    examples = []
    phases = set()
    import time
    t0 = time.time()
    for w in range(walks):
        if w % 10 == 0:
            print('  진행 %d/%d  %.0fs  불일치 %d' % (w, walks, time.time() - t0, stats['mismatch']), flush=True)
        h = Harness(rng)
        for k in range(steps):
            guided = None
            if rng.random() < 0.5:      # 절반은 모델에서 지금 가능한 사건 가운데서 고른다(깊은 단계까지 가도록)
                nx = succ(h.s)
                envc = [e for e, labs in ENV_LABEL.items() if any(l in nx for l in labs) and e != 'safe_flip']
                ctrlc = [c for c, labs in MODEL_LABEL.items() if c != 'gate_check' and any(l in nx for l in labs)]
                if ctrlc and rng.random() < 0.7:      # 제어기 단계가 가능하면 우선
                    guided = rng.choice(ctrlc)
                elif envc:
                    guided = rng.choice(envc)
            pick_env = (guided in ENV) if guided else (rng.random() < 0.55)
            if pick_env:
                e = guided or rng.choice(ENV)
                if h.env(e):
                    stats['env'] += 1
                    h.log.append(e)
            else:
                c = guided or rng.choice(CTRL)
                agree, mok, iok, why = h.ctrl(c)
                stats['ctrl'] += 1
                by_ctrl[c][0] += 1
                by_ctrl[c][1] += int(mok)
                if why:
                    reasons[why] = reasons.get(why, 0) + 1
                h.log.append('%s:%s/%s%s' % (c, 'M' if mok else 'm', 'I' if iok else 'i', '(' + why + ')' if why else ''))
                if not agree:
                    stats['mismatch'] += 1; by_ctrl[c][2] += 1
                    if len(examples) < 8:
                        examples.append((w, k, c, mok, iok, why, h.log[-8:]))
                elif mok and iok:
                    stats['ctrl_ok_both'] += 1
                else:
                    stats['ctrl_refused_both'] += 1
            phases.add(h.p.phase)
            if h.proj_impl() != h.proj_model():
                stats['proj_mismatch'] += 1
                if len(examples) < 8:
                    examples.append((w, k, 'PROJ', h.proj_model(), h.proj_impl(), None, h.log[-8:]))
                break
    print('걸음 수', walks, '×', steps, ' 환경 사건', stats['env'], ' 제어기 시도', stats['ctrl'])
    print('둘 다 허용', stats['ctrl_ok_both'], ' 둘 다 거부', stats['ctrl_refused_both'],
          ' 판단 불일치', stats['mismatch'], ' 상태 불일치', stats['proj_mismatch'])
    print('제어기 시도별 [시도, 모델 허용, 불일치]:', by_ctrl)
    print('구현의 거부 사유:', dict(sorted(reasons.items(), key=lambda x: -x[1])))
    print('도달한 단계:', sorted(phases))
    for ex in examples:
        print('예:', ex)
    return stats


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    run(walks=n)
