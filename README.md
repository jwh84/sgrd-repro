# SGRD — reproduction and verification code

<https://github.com/jwh84/sgrd-repro>

Code and machine-readable results for **Safety-Gated Reversible Denial:
Recoverable Anti-Tampering for Crewless Merchant Ships** (안전상태 게이트형 가역적
무력화: 무승무원 상선을 위한 복구 가능한 안티탬퍼).

One command reproduces every quantitative and formal claim in the paper from the
published parameterisation and diffs the recomputed values against the printed
ones.

```bash
pip install -r requirements.txt
python run_all.py
```

Expected output ends with `132/132 published values reproduced`. Add `--strict`
to make any discrepancy an exit code, `--no-figures` to skip matplotlib.

## What is here

| Path | Contents |
|---|---|
| `sgrd/modelcheck.py` | Transition system, twelve properties, five designs (B0, B1, B2, B3, H1). Exhaustive enumeration of the reachable transition relation. |
| `sgrd/ctmc.py` | The absorbing continuous-time Markov chain, solved by fundamental matrix. |
| `sgrd/threshold.py` | Poisson-binomial dynamic programme over heterogeneous holders; quorum timing. |
| `sgrd/asam.py` | Possession-duration extraction and the per-observation audit pipeline. |
| `sgrd/params.py` | Every input parameter and every published value, in one place, for diffing. |
| `sgrd/sensitivity.py` | Robustness of the findings to holder availability and compromise. |
| `sgrd/report.py`, `figures.py`, `opconcept.py` | Recompute-and-diff harness; figures. |
| `run_all.py` | Single-command reproduction. |
| `dump_counterexamples.py` | Writes one concrete violating transition per failed property, per design. |
| `results/results.json` | All computed values, machine-readable. |
| `results/counterexamples.{json,txt}` | Per-design counterexample list — the transitions that break each claim. |
| `results/asam-audit.csv` | The 16 retained observations and the 16 removals, with the rule that produced each. |
| `results/figures/` | Regenerated figures. |
| `tests/test_sgrd.py` | Invariants, reproduction, and the three findings stated as predicates. |
| `sgrd/refimpl/` | Reference implementation: the group, threshold proxy re-encryption with share-validity proofs, the recovery exchange, and the staged controller. |
| `sgrd/refimpl/store.py` | Non-volatile capsule storage and the commit protocol, with a device that can be made to lose power at a named instant. |
| `sgrd/refimpl/tpm.py`, `pcp.py` | TPM 2.0 over Windows TBS, and TPM-resident keys through the Platform Crypto Provider. Both in ctypes, no external TSS. |
| `run_refimpl.py` | Experiments S1–S6 and A1–A4 (tier 0), T1–T5 (tier 1). |
| `results/refimpl.{json,txt}` | Every check and timing from that run. |
| `tests/test_refimpl.py` | Constants, primitives, and S1–S5/T1–T5 as regression tests. |

```bash
python tests/test_sgrd.py     # 34 tests, no pytest needed
pytest -q                     # if you have it
python dump_counterexamples.py --limit 3
```

## Reference implementation

`run_all.py` checks the controller's *logic* over an abstract transition system
in which authority is one of four strings. `run_refimpl.py` checks the same
controller where those values have to be produced by real signatures, a real
transcript hash, a real monotonic counter, and a real manifest comparison.

```bash
python run_refimpl.py --strict     # ~3 minutes, 85 checks
python tests/test_refimpl.py       # 43 tests, no pytest needed
```

**Tier 0 (S1–S6)** needs nothing but an interpreter. Every check asserts the
*reason* for a refusal, not merely that one occurred. The one experiment that
measures rather than refuses is the latency budget, and its point is negative:
the whole cryptographic exchange runs four and a half orders of magnitude below
the quorum-gathering time that dominates the paper's 23-hour mean restoration,
so no plausible change to it moves any quantitative claim.

**Crash consistency (A1–A4)** covers the capsule commit. Power is cut before
every device operation, under every subset of pending writes that might have
reached the medium anyway — so reordering is covered, not just loss. A2 found a
real defect in the first version of the protocol: it chose the counter record by
epoch parity, overwriting the only valid record every second commit, and a write
torn there destroyed the store. A3 is an ablation: two of the three syncs are
load-bearing and the third is not, which is worth knowing and is not visible by
inspection. A4 kills a real process writing real files and checks the outcome
against what the simulation predicted.

**Tier 1 (T1–T5)** needs a TPM 2.0, which every machine meeting the Windows 11
installation requirement has. It is skipped, not failed, where there is none
(`--no-tpm` skips it explicitly). The result worth reading is T3: a secret
sealed to the platform's measured state is released by the chip only in that
state, and a difference in one of the eight boot PCRs is enough for it to
refuse. That is what R1 does, enforced by hardware rather than by the
controller.

Tier 1 changes no persistent state. Keys are created under the TPM's null
hierarchy, CNG keys are deleted on exit, no NV index is defined, no ownership is
taken, and no PCR is written — Windows blocks PCR writes from user mode in any
case, which is why T3 seals to *differing* states rather than mutating one.

**What is still not established.** Zeroisation completeness, PUF error rates,
the behaviour of the storage *medium* under power loss, and memory residue after
erasure. A refusal in software is a refusal by the specified logic, not by a
module on a ship; a TPM is an analogue of the components the design calls for,
not those components — it is not a PUF, its PCRs measure the boot chain rather
than the autonomy stack, and its clock is not the design's counter; and A1–A4
settle the commit algorithm, not the device it runs on.
`store.MEDIUM_REQUIREMENTS` states what the algorithm needs of that device, in
a form that can be checked against a datasheet.

`REPRODUCTION.md` documents the chain's generator, the three load-bearing
modelling choices, and the known gaps. **Read it before citing any number from
here** — several results are conditional in ways the summary table does not show.

## Scope of the verification claim

The transition system models the **online** restoration path. The accurate claim
is that H1 satisfies all twelve properties *in a specification model covering
that path*. Four threats specific to the offline boarding-team path are not in
the model: theft or cloning of a pre-authorised offline credential, expiry and
revocation of the custody attestation, the interval during which a platform with
no conduit cannot learn of a revocation, and compromise of the custody
attestation issuer itself.

Reachable state and transition counts are encoding-dependent and are reported
for information. The verdicts are the claim.

## Data

The ASAM database is public: <https://msi.nga.mil/Piracy>.

- **Use the ArcGIS feature-service export, not the shapefile.** The shapefile
  truncates the narrative field at 254 characters, which deletes the sentence a
  release date normally sits in.
- SHA-256 of the export analysed for the paper:
  `67d38f66747f1ae15f2bbd05cf4b0da217568c8c89d05018ada1ea9aa5cc3c77`
  (retrieved 2026-07-30).
- Re-run the derivation against your own download with
  `python run_all.py --asam-csv <path>`. It rewrites `results/asam-audit.csv`
  with a SHA-256 of whatever it ran over.

**Do not commit the ASAM export to this repository.** ASAM narratives are not
redistributed here; `results/asam-audit.csv` carries only reference numbers,
dates, derived durations, the basis for each, stratum, and inclusion. The
reference numbers locate each row in the published export. `.gitignore` excludes
`*.csv` at the repository root and `data/` for this reason.

No proprietary or restricted data is used.

## Citation

See `CITATION.cff`. The paper's DOI and journal details are placeholders until
publication — update them before announcing the repository.

## License

Two licences, split by what the file is.

| Scope | Licence | File |
|---|---|---|
| Software — `run_all.py`, `dump_counterexamples.py`, `sgrd/`, `tests/` | MIT | `LICENSE` |
| Derived data and results — everything under `results/` | CC BY 4.0 | `results/LICENSE` |

The underlying ASAM records are published by the US National
Geospatial-Intelligence Agency and are **not** redistributed here; neither
licence grants any right over them. Obtain them from
<https://msi.nga.mil/Piracy>.

Copyright is asserted for the three authors. If Gachon University or LIG
Defense & Aerospace holds copyright in the work instead, change the holder line
in both licence files before publishing.
