# Reproduction notes

> Table and section numbers below refer to the **English manuscript**. The
> Korean version (융합보안논문지) renumbers them: Table 3 → 표 1, Table 6 → 표 2,
> Table 7 → 표 3, §VI-D → 5.3절, §VII-B → 6.2절.

Reproduces every quantitative and formal claim in *Safety-Gated Reversible
Denial: Recoverable Anti-Tampering for Crewless Merchant Ships* from the
published parameterisation, and diffs the recomputed values against the ones the
paper prints.

```bash
python run_all.py
```

Two dependencies (`numpy`, `matplotlib`); `--no-figures` drops the second.

## What it checks

| Paper | Reproduced by | Notes |
|---|---|---|
| Table 3 (ASAM possession duration) | `sgrd/asam.py` | summary stats as printed; `--asam-csv` re-runs the extraction and audit over the raw NGA export and writes `results/asam-audit.csv` |
| Table 6 (property verification) | `sgrd/modelcheck.py` | exhaustive enumeration of the reachable transition relation over five designs: B0, B1, B2, B3, H1 |
| Table 6 mechanism column | `sgrd/modelcheck.py` | `ablate()` flips each guard in turn; seven of ten are load-bearing |
| Table 7 (outcomes by configuration) | `sgrd/ctmc.py` | absorbing CTMC, solved by fundamental matrix |
| §VII-B (extraction by regime) | `sgrd/ctmc.py` | closed form, cross-checked against the matrix solve |
| Findings 1–3 | `sgrd/report.py` | stated as falsifiable predicates, not prose |
| Figs. 3–4 | `sgrd/figures.py` | regenerated from the computed values |

Twelve properties, not ten: P11 and P12 (navigation survives denial; command is
never held under possession) were added after the safety half of Requirement 3
turned out to be unexpressible in a safe-state Boolean. B0 satisfies 1 of 12 and
H1 all 12; B1/B2/B3 each hold all of SGRD but one contribution and fail exactly
the properties that contribution discharges.

`run_all.py` prints a per-value `OK`/`FAIL` line for each published number.
`--strict` makes a discrepancy an exit code.

## The chain, stated explicitly

The paper describes the CTMC in prose; the reproduction had to commit to a
specific generator. Transient states `Z_p, Z_r, Z_u, RS, RM`, absorbing
`OK, EXT, BRK`, with rates:

```
Z_p --1/T_ext-->        EXT          extraction needs the hardware (B0 only)
Z_p --1/T_poss * p_a--> Z_r          custody back, quorum assemblable
Z_p --1/T_poss *(1-p_a)-> Z_u        custody back, no quorum
Z_r --lambda_g-->       RC           quorum gathered
Z_u --1/720-->          BRK
RC  --mu*(1-p_mod)-->   OK
RC  --mu*p_mod-->       RM           R1 attestation mismatch (H1 only)
RM  --1/24-->           OK
Z_r,Z_u,RC,RM --1/720--> BRK         write-off competes AFTER custody only
```

Three choices are load-bearing and are stated in the paper rather than inferred:

1. **The write-off hazard does not compete from `Z_p`.** A constructive total
   loss is an accounting event; declaring it does not end the adversary's
   physical access, so it cannot truncate an extraction in progress. The ASAM
   durations are seizure-to-release intervals and so already conditioned on
   release, which is what makes regaining custody the modelled end of
   possession. An earlier draft did let write-off exit `Z_p`, which suppressed
   P(ext) for both variants and by different factors.
2. **`RM -> OK` directly, not `RM -> RC`.** Remediation returns a clean
   configuration straight to capability.
3. **B0 has `p_mod = 0`.** With no post-possession attestation, B0 never routes
   to remediation -- it restores the tampered configuration silently. This is a
   *flattering* choice for B0: it scores a silently restored tampered stack as a
   success. Finding 3 is about exactly that kind of artefact.

**Only B0 is priced for extraction.** H1's resistance is conditional on evidence
that the module's zeroisation is complete, so the paper rates it in an
attack-potential band instead (Section VI-D); `ctmc.extraction_probability`
raises `KeyError` if asked to price it, and `evaluate()` returns `None` for
`p_extracted` and for both unconditional figures. Everything conditional on
custody is still computed, because those quantities are entered at `Z_r`/`Z_u`
and never read `Z_p`'s row.

## Known gaps

- **Reachable-state counts are encoding-dependent.** Table 6 now reports the
  counts this encoding produces — 722 states / 3116 transitions for B0 and
  346 / 1430 for H1 — but any count of that kind is a property of how the
  transition system is variable-encoded, not of the semantics. An earlier draft
  of the paper printed 1536 / 9684 and 648 / 3168 from a coarser encoding. The
  verdicts — B0 satisfies 1 of 12, H1 all 12, and it is P9 that B0 gets right —
  are encoding-robust, and those are the claim. `run_all.py` dumps one concrete
  violating transition per failed property, so the B0 failures can be read
  rather than counted.
- **The 16 audited ASAM observations are published** as `results/asam-audit.csv`
  (reference, dates, duration, its basis, stratum, inclusion, exclusion rule),
  alongside the summary statistics. The quantitative model needs only the stratum means, which *are*
  published, so the reproduction is complete without them; the derivation step
  is re-executed against a fresh NGA export with `--asam-csv`, and writes a
  per-observation audit trail so the reduction can be argued row by row.
- **Take the ArcGIS feature service, not the shapefile.** `ASAM_events_V1`
  serves the full narrative; the shapefile truncates it at 254 characters,
  which deletes the sentence a release date normally sits in and reduces the
  recoverable sample to almost nothing.
- **Thirteen of the sixteen durations take their release year from the record
  rather than from the sentence.** ASAM narratives usually give a bare day and
  month ("released on 18 July"); the record carries the year. That reading is
  applied explicitly, rolled forward by a year where the release would
  otherwise precede the seizure, and declared per observation in the `basis`
  column of the audit trail. A reader who rejects it can drop exactly those
  rows -- which is the point of recording it.
- **The chain collapses the gather stage to one exponential.** Time-to-quorum is
  really a mixture of sums of exponentials. Matching its mean preserves E[T_c]
  exactly, but *not* the absorption probabilities: with a competing write-off
  hazard the split is E[exp(-lambda_wo T)], which depends on the whole
  distribution of T and not on its mean. The error is small here because
  lambda_wo * E[T] is small, and `--sensitivity` bounds it, but the earlier
  claim that the reduction preserved every absorption probability was wrong.
  The collapse also understates the variance,
  so this code cannot answer questions about the *tail* of restoration latency.

## Layout

```
.
  run_all.py               single-command reproduction
  dump_counterexamples.py  per-design violating transitions
  sgrd/
    params.py         Table 3 + every published value, for diffing
    threshold.py      Poisson-binomial DP; quorum timing
    ctmc.py           the absorbing chain
    modelcheck.py     transition system, twelve properties, five designs
    asam.py           possession duration + audit pipeline
    report.py         recompute-and-diff harness
    figures.py        Figs. 3-4
  tests/test_sgrd.py  invariants, reproduction, and the findings as predicates
```

```bash
python tests/test_sgrd.py     # no pytest needed
pytest -q                     # if you have it
```

## Data

The ASAM database is public: <https://msi.nga.mil/Piracy>. Download the bulk CSV
and pass `--asam-csv <path>` to re-run the derivation. Nothing here requires
proprietary or restricted data.
