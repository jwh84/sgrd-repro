"""Possession duration from the NGA Anti-Shipping Activity Messages database
(Section `sec:asam`).

The rate at which custody is regained sets the length of the adversary's working
window, and every extraction result turns on it. The paper derives it rather
than stipulating it, from the public ASAM bulk export (9182 records, 1978-2024).

Two entry points:

* :func:`published_table` returns Table `tab:asam` as printed. The Markov chain
  needs only the stratum means, which are published, so the whole quantitative
  reproduction runs from this without the raw export.
* :func:`derive_from_export` runs the actual extraction and audit pipeline over
  a downloaded ASAM CSV, so the derivation itself can be re-executed rather than
  taken on trust. Get the export from https://msi.nga.mil/Piracy.

The raw sixteen audited observations are not printed in the paper, so
:func:`published_table` is a table of summary statistics, not of data. Anything
that needs the individual durations must go through :func:`derive_from_export`.

The export to use is the ArcGIS feature service NGA publishes behind that page,
``ASAM_events_V1``; its ``description`` field carries the full narrative, where
the shapefile's does not. That matters: the shapefile truncates at 254
characters and loses the sentence a release date usually sits in.
"""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean, median

# --- Published summary (Table tab:asam) --------------------------------------

#: Two strata, not four. The audited sample places ten of sixteen observations
#: in NAVAREA II; the remaining six are spread over three NAVAREAs at n = 1, 2
#: and 3, which will not carry a stratum mean. Splitting them further would
#: report as regional structure what is one observation.
STRATUM_ORDER = (
    "Pooled",
    "NAVAREA II (Gulf of Guinea)",
    "Other NAVAREAs (VIII, IX, XI)",
)


@dataclass(frozen=True)
class Stratum:
    name: str
    n: int
    median_h: float
    mean_h: float
    p90_h: float
    max_h: float


def published_table() -> list[Stratum]:
    """Table `tab:asam` exactly as printed, in hours."""
    return [
        Stratum("Pooled", 16, 120.0, 175.5, 444.0, 648.0),
        Stratum("NAVAREA II (Gulf of Guinea)", 10, 84.0, 86.4, 146.4, 168.0),
        Stratum("Other NAVAREAs (VIII, IX, XI)", 6, 228.0, 324.0, 624.0, 648.0),
    ]


def check_internal_consistency() -> list[str]:
    """Cheap sanity checks on the published table.

    The strata partition the pooled sample, so the counts must add up and the
    pooled mean must be the count-weighted mean of the strata. This catches
    transcription errors without needing the raw data.
    """
    rows = {s.name: s for s in published_table()}
    pooled = rows["Pooled"]
    strata = [s for s in published_table() if s.name != "Pooled"]
    problems: list[str] = []

    total_n = sum(s.n for s in strata)
    if total_n != pooled.n:
        problems.append(f"stratum counts sum to {total_n}, pooled n is {pooled.n}")

    weighted = sum(s.n * s.mean_h for s in strata) / total_n
    if abs(weighted - pooled.mean_h) > 1.0:
        problems.append(
            f"count-weighted stratum mean is {weighted:.1f} h, "
            f"pooled mean is printed as {pooled.mean_h:.1f} h")

    if max(s.max_h for s in strata) != pooled.max_h:
        problems.append("pooled max does not equal the largest stratum max")

    if min(s.n for s in strata) < 5:
        problems.append(
            f"a stratum has n = {min(s.n for s in strata)}; a mean is not "
            f"reported for fewer than five observations")

    return problems


# --- Derivation from the raw export ------------------------------------------

#: ASAM's actor field names the perpetrator, not the outcome, so possession-type
#: events are identified from the narrative. These are the cues for "the
#: adversary took or retained control of the platform".
POSSESSION_CUES = (
    r"\bhijack", r"\bkidnap", r"\bseiz", r"\bcaptur", r"\btook control",
    r"\btaken control", r"\bheld\b", r"\bdetain", r"\babduct",
    r"\bunder (?:the )?control of", r"\bcommandeer",
)
_POSSESSION_RE = re.compile("|".join(POSSESSION_CUES), re.IGNORECASE)

# --- Reading a possession duration out of a narrative ------------------------
#
# ASAM narratives state a release three ways, and an extractor that accepts only
# the tidiest of them recovers almost nothing: of 1267 possession events, two
# carry a release date complete with its year. What the rest carry is a bare day
# and month ("the vessel was released on 18 July"), or the duration in words
# ("released after 19 days").
#
# Reading the year off the record rather than out of the sentence is the one
# inference made here, and it is declared per observation in
# ``Observation.basis`` so that a reader can discount those rows. It is a
# reading rule and not an estimate: an ASAM record is a single-incident report
# carrying its own occurrence date, and a release given as a bare day and month
# belongs to that year unless it would fall before the seizure, in which case it
# belongs to the next.

#: The release event. ``return`` is admitted only as "returned to <someone>":
#: bare "returned" matches "the patrol boat returned fire", which is not one.
_EVENT = (r"(?:releas\w*|freed|recover\w*|let\s+go|hand(?:ed)?\s+back"
          r"|returned\s+to)")
_RE_EVENT = re.compile(_EVENT, re.IGNORECASE)

_CREW = (r"crew\w*|crewmen|hostage\w*|seafarer\w*|sailor\w*|master"
         r"|mariner\w*|personnel")
_VESSEL = (r"vessel|ship|tanker|yacht|boat|barge|tug|dhow|trawler|freighter"
           r"|carrier|craft")

#: Release date with its year stated, day-first and month-first.
_RE_DMY_Y = re.compile(
    rf"{_EVENT}[^.]{{0,90}}?\bon\s+(\d{{1,2}})\s+([A-Za-z]{{3,9}})\.?,?\s+"
    rf"(\d{{4}}|\d{{2}})\b", re.IGNORECASE)
_RE_MDY_Y = re.compile(
    rf"{_EVENT}[^.]{{0,90}}?\bon\s+([A-Za-z]{{3,9}})\.?\s+(\d{{1,2}}),?\s+"
    rf"(\d{{4}}|\d{{2}})\b", re.IGNORECASE)
#: The same with no year, where the year is read from the record.
_RE_DM_NOY = re.compile(
    rf"{_EVENT}[^.]{{0,90}}?\bon\s+(\d{{1,2}})\s+([A-Za-z]{{3,9}})\b"
    rf"(?!\s*,?\s*\d)", re.IGNORECASE)
_RE_MD_NOY = re.compile(
    rf"{_EVENT}[^.]{{0,90}}?\bon\s+([A-Za-z]{{3,9}})\.?\s+(\d{{1,2}})\b"
    rf"(?!\s*,?\s*\d)", re.IGNORECASE)
#: The duration stated outright, which needs no release date at all.
_RE_AFTER = re.compile(
    rf"{_EVENT}[^.]{{0,60}}?(?:after|following)\s+(\w+)\s+"
    rf"(day|week|month)s?", re.IGNORECASE)
_RE_HELD = re.compile(
    rf"(?:held|detained|hijacked|captiv\w*)[^.]{{0,70}}?(?:for\s+)?(\w+)\s+"
    rf"(day|week|month)s?[^.]{{0,50}}?{_EVENT}", re.IGNORECASE)

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
             "twelve": 12}
_UNIT_H = {"day": 24.0, "week": 168.0, "month": 720.0}

#: Occurrence-date formats, as the ArcGIS export and the web export serve them.
_DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y")

#: A release that is asked for, refused or negotiated is not a release.
_RE_UNREAL = re.compile(
    r"will not|would not|refus\w+|demand\w*|negotiat\w*|"
    r"ransom for the release|seeking the release|calls? for the release|"
    r"appeal\w* for", re.IGNORECASE)
#: An attack driven off never produced a possession to end.
_RE_REPELLED = re.compile(
    r"broke away|thwart\w*|repel\w*|unsuccessful|fled (?:the|after)|"
    r"attempt\w* to board|prevented the (?:boarding|attack)|drove (?:them )?off",
    re.IGNORECASE)
_RE_HIJACK_CONFIRMED = re.compile(
    r"hijack\w*ed the|seized the|took control", re.IGNORECASE)
_RE_CENSORED = re.compile(
    r"still (?:being )?held|remains? (?:in )?captiv|not(?: yet)? been released",
    re.IGNORECASE)
#: "crew taken from the <vessel>" names where they were taken from, not what was
#: released; without this the vessel noun sits closer to the verb and wins.
_RE_TAKEN_FROM = re.compile(
    rf"(?:{_CREW})[^.]{{0,50}}?(?:taken|kidnapp\w*|abduct\w*|seiz\w*)"
    rf"[^.]{{0,40}}?\bfrom\b", re.IGNORECASE)


def _release_year(token: str) -> int:
    y = int(token)
    return y if y >= 1000 else (2000 + y if y < 50 else 1900 + y)


def _subject_of_release(text: str, at: int) -> str | None:
    """Whether the crew or the platform is what the narrative says was released.

    Decided at the verb the duration was read from, not over the whole
    narrative: a report can mention both, and only one of them is the thing
    whose possession ended.
    """
    if _RE_TAKEN_FROM.search(text):
        return "crew"
    m = _RE_EVENT.search(text, at)
    if not m:
        return None
    window = text[max(0, m.start() - 120):m.start()]
    crew = [x.end() for x in re.finditer(_CREW, window, re.IGNORECASE)]
    vessel = [x.end() for x in re.finditer(_VESSEL, window, re.IGNORECASE)]
    if not crew and not vessel:
        return None
    if not vessel:
        return "crew"
    if not crew:
        return "vessel"
    return "crew" if crew[-1] > vessel[-1] else "vessel"


def _read_duration(text: str, occurred: date) -> tuple[float | None, str | None]:
    """Return ``(hours, basis)``; ``basis`` names what the value rests on."""
    for rx, day_first in ((_RE_DMY_Y, True), (_RE_MDY_Y, False)):
        m = rx.search(text)
        if not m:
            continue
        d, mon, y = ((m.group(1), m.group(2), m.group(3)) if day_first
                     else (m.group(2), m.group(1), m.group(3)))
        mi = _MONTHS.get(mon[:3].lower())
        if mi:
            try:
                rel = date(_release_year(y), mi, int(d))
            except ValueError:
                continue
            return ((rel - occurred).days * 24.0 or SAME_DAY_HOURS,
                    "release date stated in full")
    for rx, day_first in ((_RE_DM_NOY, True), (_RE_MD_NOY, False)):
        m = rx.search(text)
        if not m:
            continue
        d, mon = ((m.group(1), m.group(2)) if day_first
                  else (m.group(2), m.group(1)))
        mi = _MONTHS.get(mon[:3].lower())
        if not mi:
            continue
        try:
            rel = date(occurred.year, mi, int(d))
        except ValueError:
            continue
        basis = "year read from the record"
        if rel < occurred:
            try:
                rel = rel.replace(year=rel.year + 1)
            except ValueError:
                continue
            basis = "year read from the record, rolled forward"
        return (rel - occurred).days * 24.0 or SAME_DAY_HOURS, basis
    for rx in (_RE_AFTER, _RE_HELD):
        m = rx.search(text)
        if m:
            tok = m.group(1).lower()
            n = int(tok) if tok.isdigit() else _WORD_NUM.get(tok)
            if n:
                return n * _UNIT_H[m.group(2).lower()], "duration stated in words"
    return None, None


def _duration_site(text: str) -> int:
    """Offset of the release verb the duration was read from."""
    for rx in (_RE_DMY_Y, _RE_MDY_Y, _RE_DM_NOY, _RE_MD_NOY, _RE_AFTER, _RE_HELD):
        m = rx.search(text)
        if m:
            return m.start()
    return 0


#: ASAM resolves dates to the day, so a same-day release is recorded as this
#: many hours rather than as zero.
SAME_DAY_HOURS = 12.0


@dataclass(frozen=True)
class Observation:
    reference: str
    nav_area: str
    occurred: date
    released: date
    duration_h: float
    narrative: str
    #: What the duration rests on: a release date stated in full, a year read
    #: from the record, or a duration stated in words. Carried into the audit
    #: trail so that rows resting on the reading rule can be discounted.
    basis: str = ""

    @property
    def stratum(self) -> str:
        """One of the two strata of :data:`STRATUM_ORDER`, by NAVAREA."""
        return (STRATUM_ORDER[1] if self.nav_area.strip().upper() == "II"
                else STRATUM_ORDER[2])


@dataclass(frozen=True)
class AuditResult:
    """The audit trail of Section `sec:asam`: 32 candidates, 16 removed, 16 kept."""

    n_records: int
    n_possession_events: int
    candidates: list[Observation]
    kept: list[Observation]
    removed: list[tuple[Observation, str]]


def write_audit_csv(result: AuditResult, out_path: str | Path,
                    source_path: str | Path | None = None) -> Path:
    """Emit the per-observation audit trail behind Table `tab:asam`.

    Section `sec:asam` reports 32 candidates reduced to 16 by an explicit audit,
    but the paper has room only for the counts. This writes the row-level record
    a reader would need to contest a specific inclusion: the reference, the two
    dates, the derived duration, *what the duration rests on*, the stratum,
    whether the observation was kept, and for a removal the rule that removed
    it. The basis column is the one to read first: thirteen of the sixteen
    durations rest on a year taken from the record rather than from the
    sentence, and a reader who rejects that rule can drop exactly those rows.

    Narratives are not copied. Reproducing them wholesale would republish the
    source rather than document the audit; the reference number locates each row
    in the published export instead. ``source_path``, when given, is hashed so that the
    export the audit ran over can be identified exactly.
    """
    out = Path(out_path)
    digest = None
    if source_path is not None:
        h = hashlib.sha256()
        with Path(source_path).open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        digest = h.hexdigest()

    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([f"# ASAM audit trail; source records {result.n_records}, "
                    f"possession-type events {result.n_possession_events}, "
                    f"candidates {len(result.candidates)}, kept {len(result.kept)}"])
        if digest:
            w.writerow([f"# source sha256 {digest}"])
        w.writerow(["reference", "nav_area", "stratum", "occurred", "released",
                    "duration_h", "basis", "included", "exclusion_reason"])
        kept = {id(o) for o in result.kept}
        reasons = {id(o): r for o, r in result.removed}
        for obs in result.candidates:
            w.writerow([obs.reference, obs.nav_area, obs.stratum,
                        obs.occurred.isoformat(),
                        obs.released.isoformat() if obs.released else "",
                        f"{obs.duration_h:.1f}" if obs.duration_h else "",
                        obs.basis,
                        "yes" if id(obs) in kept else "no",
                        reasons.get(id(obs), "")])
    return out


def _parse_date(text: str) -> date | None:
    text = text.strip().replace(",", "")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _column(header: list[str], *candidates: str) -> str | None:
    lowered = {h.lower().replace("_", "").replace(" ", ""): h for h in header}
    for c in candidates:
        key = c.lower().replace("_", "").replace(" ", "")
        if key in lowered:
            return lowered[key]
    return None


def derive_from_export(csv_path: str | Path) -> AuditResult:
    """Run the extraction and audit over a downloaded ASAM CSV.

    The five documented removals are reproduced as explicit rules rather than as
    a hand-edited list: a year-only placeholder date, a report date mistaken for
    a seizure date (release before occurrence), right-censored cases where the
    platform was still held, and cases where only the crew was released.
    """
    path = Path(csv_path)
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        header = list(reader.fieldnames)
        # `dateofocc` and `subreg` are what the ArcGIS feature service actually
        # serves; the longer names are what the web export and the shapefile
        # carry. Accept both rather than make the caller rename columns.
        col_date = _column(header, "dateOfOccurrence", "dateofocc", "date",
                           "occurrenceDate")
        col_text = _column(header, "description", "narrative", "text", "remarks")
        col_area = _column(header, "navArea", "navarea", "area", "subreg")
        col_ref = _column(header, "reference", "referenceNumber", "id")
        missing = [n for n, c in (("date", col_date), ("description", col_text))
                   if c is None]
        if missing:
            raise ValueError(
                f"{path} is missing required column(s) {missing}; header was {header}")
        rows = list(reader)

    possession: list[dict[str, str]] = []
    for row in rows:
        text = row.get(col_text) or ""
        if _POSSESSION_RE.search(text):
            possession.append(row)

    candidates: list[Observation] = []
    for row in possession:
        text = row.get(col_text) or ""
        occurred_raw = (row.get(col_date) or "").strip()
        occurred = _parse_date(occurred_raw[:10]) if len(occurred_raw) >= 8 else None
        ref = (row.get(col_ref) or "?") if col_ref else "?"
        area = (row.get(col_area) or "?") if col_area else "?"
        if occurred is None:
            continue
        hours, basis = _read_duration(text, occurred)
        if hours is None:
            continue
        released = occurred + timedelta(hours=hours) if hours >= 0 else date.min
        # A 1 January occurrence is ASAM's year-only placeholder, so the
        # interval it would give is a year-scale artefact rather than a
        # measurement. Flagged here and removed by the audit below.
        if occurred_raw[5:10] == "01-01":
            occurred = date.min
        candidates.append(
            Observation(ref, area, occurred,
                        released if isinstance(released, date) else date.min,
                        hours, text, basis))

    kept: list[Observation] = []
    removed: list[tuple[Observation, str]] = []
    for obs in candidates:
        text = obs.narrative
        if obs.occurred == date.min:
            removed.append((obs, "year-only placeholder occurrence date"))
        elif obs.duration_h < 0:
            removed.append((obs, "release precedes occurrence: report date "
                                 "mistaken for a seizure date"))
        elif obs.duration_h > 24 * 365:
            removed.append((obs, "implausible: over one year"))
        elif _RE_CENSORED.search(text):
            removed.append((obs, "right-censored: platform still held"))
        elif _RE_UNREAL.search(text):
            removed.append((obs, "release is demanded, refused or under "
                                 "negotiation, not reported as having happened"))
        elif _RE_REPELLED.search(text) and not _RE_HIJACK_CONFIRMED.search(text):
            removed.append((obs, "attack repelled: the platform was never "
                                 "possessed, so no possession ended"))
        elif _subject_of_release(text, _duration_site(text)) == "crew":
            removed.append((obs, "the crew was released, not the platform"))
        else:
            kept.append(obs)

    return AuditResult(len(rows), len(possession), candidates, kept, removed)


def summarise(observations: list[Observation], name: str) -> Stratum:
    """Reduce a set of observations to the columns of Table `tab:asam`."""
    if not observations:
        raise ValueError(f"stratum {name!r} is empty")
    durations = sorted(o.duration_h for o in observations)
    return Stratum(
        name=name,
        n=len(durations),
        median_h=float(median(durations)),
        mean_h=float(mean(durations)),
        p90_h=_percentile(durations, 0.90),
        max_h=durations[-1],
    )


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile, matching numpy's default."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (pos - lo) * (sorted_values[hi] - sorted_values[lo])


#: The four caveats of Section `sec:asam`, carried in code so that any consumer
#: of these numbers has to see them.
CAVEATS = (
    "Use the feature-service export, not the shapefile: the shapefile's "
    "narrative field is truncated at 254 characters, which removes the "
    "sentence a release is usually stated in.",
    "The sample is not a random draw from possession events but from the "
    "minority whose narrative records an ending, and the two selection biases "
    "run opposite ways: long detentions attract follow-up reporting, while the "
    "longest-running episodes are still unresolved when the report is filed "
    "and are removed as right-censored.",
    "ASAM resolves dates to the day, so a same-day release is recorded as "
    f"{SAME_DAY_HOURS:.0f} hours.",
    "The audited sample is 16 observations, thirteen of which take their "
    "release year from the record rather than the sentence: usable for order "
    "of magnitude and central tendency, not as a fitted distribution.",
)
