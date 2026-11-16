# -*- coding: utf-8 -*-
"""Monte Carlo cross-check of the CTMC (review items M3, 5.4.3). 정량 모델 몬테카를로 교차 검증.

국면 2(관리권 회복 뒤): 사건 단위로 한 회차를 그대로 흉내 낸다.
  - 보유자 i 는 확률 a_i 로 참여하고, 참여한 보유자는 Exp(rho) 뒤에 응답한다. 응답이 t 개 모이면 정족수가 모인다.
    (체인은 이 시간을 평균이 같은 지수분포 하나로 근사한다. 시뮬레이션은 근사 없이 t 번째 응답 시각을 쓴다.)
  - 단계적 복원, 확률 p_mod 로 선외 조치 → 정족수 재조립(같은 참여자, 새 응답 시간) → 재복원.
  - 손실처리 기한까지 복원이 끝나지 않으면 손실.
  분포 설정을 바꾸어 비지수 가정에서도 결론(구성 순서, 발견 1)이 유지되는지 본다.
국면 1(B0): 점유 지속시간을 지수 대신 ASAM 16건에 최대우도로 맞춘 로그정규로, 추출 시간을 고정값으로 바꾼다.
"""
import math
import os
import sys
import numpy as np
sys.dont_write_bytecode = True
from . import params as P, ctmc
from .threshold import p_authority_assemblable, quorum_rate

RNG = np.random.default_rng(20261114)
N = 400_000


def draw(kind, mean, n):
    if kind == 'exp':
        return RNG.exponential(mean, n)
    if kind == 'det':
        return np.full(n, float(mean))
    if kind.startswith('erl'):                   # Erlang-k, 평균 유지
        k = int(kind[3:])
        return RNG.gamma(k, mean / k, n)
    raise ValueError(kind)


def gather_mech(avail, t, rho, part, n):
    """참여 행렬 part(n×m)가 주어졌을 때 t 번째 응답 시각. 정족수 불가면 inf."""
    m = len(avail)
    resp = RNG.exponential(1.0 / rho, (n, m))
    resp = np.where(part, resp, np.inf)
    resp.sort(axis=1)
    return resp[:, t - 1]


def episode(cfg, dist, n=N):
    """관리권 회복 시점부터 한 회차. 반환: P(rest|c), P(loss|c), E[T|OK]."""
    holders, t = cfg.holders, cfg.t
    a = np.array([h.availability for h in holders])
    rho = P.RESPONSE_RATE
    part = RNG.random((n, len(a))) < a
    assemblable = part.sum(axis=1) >= t
    if dist['gather'] == 'ctmc':                 # 체인과 같은 근사: 평균이 같은 지수분포
        lam = quorum_rate(holders, t, rho)
        g1 = np.where(assemblable, RNG.exponential(1 / lam, n), np.inf)
        g2 = RNG.exponential(1 / lam, n)
    else:
        g1 = gather_mech(a, t, rho, part, n)
        g2 = gather_mech(a, t, rho, part, n)
    r1 = draw(dist['restore'], 1 / P.RESTORATION_RATE['H1'], n)
    r2 = draw(dist['restore'], 1 / P.RESTORATION_RATE['H1'], n)
    rem = draw(dist['remediate'], 1 / P.REMEDIATION_RATE, n)
    mod = RNG.random(n) < P.P_MODIFIED
    total = g1 + r1 + np.where(mod, rem + g2 + r2, 0.0)
    deadline = draw(dist['deadline'], P.WRITE_OFF_HORIZON_H, n)
    ok = assemblable & (total <= deadline)
    return ok.mean(), 1 - ok.mean(), total[ok].mean()


DISTS = {
    'D0 체인과 같은 가정(검증용)': dict(gather='ctmc', restore='exp', remediate='exp', deadline='exp'),
    'D1 정족수 실제 메커니즘': dict(gather='mech', restore='exp', remediate='exp', deadline='exp'),
    'D2 + 고정 기한 30일': dict(gather='mech', restore='exp', remediate='exp', deadline='det'),
    'D3 + 단계 4개(Erlang-4)·선외 조치 고정 24시간': dict(gather='mech', restore='erl4', remediate='det', deadline='exp'),
    'D4 모두(실제 정족수, Erlang-4, 고정 선외 조치, 고정 기한)': dict(gather='mech', restore='erl4', remediate='det', deadline='det'),
}


def phase2():
    cfgs = [c for c in P.named_configurations() if c.variant == 'H1']
    print('== 국면 2: P(rest|c) / P(loss|c) / E[T|OK] (시행 %d회)' % N)
    print('  해석해(체인): ' + ' | '.join('%s %.3f/%.3f/%.1f' % (
        c.name.split(' ')[0], ctmc.evaluate(c).p_restored_given_custody, ctmc.evaluate(c).p_loss_given_custody,
        ctmc.evaluate(c).mean_time_to_restore_given_custody) for c in cfgs))
    rows = {}
    for name, d in DISTS.items():
        res = {c.name.split(' ')[0]: episode(c, d) for c in cfgs}
        rows[name] = res
        print('  %s: ' % name + ' | '.join('%s %.3f/%.3f/%.1f' % (k, *v) for k, v in res.items()))
        order = sorted(res, key=lambda k: res[k][1])
        f1 = res['2-of-4'][1] <= res['2-of-3'][1] and res['3-of-5'][1] <= res['3-of-4'][1]
        print('     손실 순서(낮은 것부터):', order, ' 발견 1(보유자 추가가 손실을 키우지 않음):', f1)
    return rows


def phase1():
    rows = [l for l in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'asam-audit.csv'), encoding='utf-8')
            if not l.lstrip('"').startswith('#')]
    import csv
    kept = [float(r['duration_h']) for r in csv.DictReader(rows) if r['included'] == 'yes']
    x = np.log(kept)
    mu, sd = x.mean(), x.std()           # 최대우도
    print('\n== 국면 1: B0 의 추출 확률 (A2 추출 72시간, A3 24시간), ASAM %d건' % len(kept))
    print('  로그정규 최대우도: mu=%.3f sigma=%.3f, 평균 %.1f h (표본 평균 %.1f h)' % (mu, sd, math.exp(mu + sd * sd / 2),
                                                                               np.mean(kept)))
    n = N
    poss = {'지수(평균 175.5h, 현재)': RNG.exponential(P.POSSESSION_POOLED_H, n),
            '로그정규(최대우도)': RNG.lognormal(mu, sd, n),
            '경험분포(16건 재표본)': RNG.choice(kept, n)}
    for t_ext, lab in ((72.0, 'A2'), (24.0, 'A3')):
        ext = {'지수': RNG.exponential(t_ext, n), '고정': np.full(n, t_ext), 'Erlang-4': RNG.gamma(4, t_ext / 4, n)}
        for pn, pv in poss.items():
            cells = ['%s %.3f' % (en, (ev < pv).mean()) for en, ev in ext.items()]
            print('  %s, 점유 %s: ' % (lab, pn) + ', '.join(cells))
    print('  해석해(지수·지수): A2 %.3f, A3 %.3f' % (72 ** -1 / (72 ** -1 + 1 / 175.5), 24 ** -1 / (24 ** -1 + 1 / 175.5)))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    phase2()
    phase1()
