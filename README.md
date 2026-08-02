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

```bash
python tests/test_sgrd.py     # 34 tests, no pytest needed
pytest -q                     # if you have it
python dump_counterexamples.py --limit 3
```

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
