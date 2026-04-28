"""Query normalization layer.

Runs before the agent. Three jobs:

  1. Cleanup: lowercase, collapse whitespace, strip trailing punct.
  2. Abbreviation expansion for the bio-imaging vocabulary the lab
     used. Operates additively — we keep the original abbreviation
     in case the agent needs it for citations.
  3. Constraint extraction: pull "at least 100 subjects", "older
     than 40", etc. into structured `hints` the agent can pass
     directly into metadata_filter without inferring them itself.

The output (NormalizedQuery) is what every other layer sees.
"""

from __future__ import annotations

import re

from datafinder.schema import NormalizedQuery

# Bio-imaging abbreviations the lab's queries used most. Order
# matters only for substring overlap (e.g. expand `OCT` before
# `CT`). We dedupe at the end so adding both forms doesn't create
# duplicates if the user already wrote the long form.
_ABBREVIATIONS: list[tuple[str, str]] = [
    ("OAI", "Osteoarthritis Initiative"),
    ("ADNI", "Alzheimer's Disease Neuroimaging Initiative"),
    ("UKBB", "UK Biobank"),
    ("OCT", "optical coherence tomography"),
    ("MRI", "magnetic resonance imaging"),
    ("fMRI", "functional magnetic resonance imaging"),
    ("dMRI", "diffusion magnetic resonance imaging"),
    ("CT", "computed tomography"),
    ("PET", "positron emission tomography"),
    ("EEG", "electroencephalography"),
    ("ECG", "electrocardiography"),
]


# Regex helpers for constraint extraction.
_RE_AT_LEAST_N_SUBJECTS = re.compile(
    r"(?:at\s+least|>=?|min(?:imum)?\s+of)\s+(\d+)\s+(?:subject|patient|participant|case)s?",
    re.I,
)
_RE_AT_LEAST_N_BARE = re.compile(
    r"(\d+)\+\s+(?:subject|patient|participant|case)s?",
    re.I,
)
_RE_OLDER_THAN = re.compile(
    r"(?:older\s+than|over|>=?\s*)\s*(\d{1,3})\s*(?:years?\s*old|y/?o|yo|years?)?",
    re.I,
)
_RE_AGE_BETWEEN = re.compile(
    # Accept hyphen-minus, en-dash (U+2013), em-dash (U+2014), or the
    # words "to" / "and" — clinicians paste age ranges from many sources.
    r"(?:age|aged)?\s*(?:between\s+)?(\d{1,3})\s*(?:-|\u2013|\u2014|to|and)\s*(\d{1,3})",
    re.I,
)
_RE_AGE_PLUS = re.compile(r"(?:age\s+)?(\d{1,3})\+", re.I)


def normalize(query: str) -> NormalizedQuery:
    raw = query
    text = " ".join(query.split())  # collapse whitespace
    text = text.rstrip(".?!;:,")

    text = _expand_abbreviations(text)
    text = " ".join(text.split())  # re-collapse after expansions

    hints: dict[str, object] = {}
    _extract_subject_count(text, hints)
    _extract_age(text, hints)

    return NormalizedQuery(raw=raw, text=text, hints=hints)


def _expand_abbreviations(text: str) -> str:
    out = text
    for abbr, full in _ABBREVIATIONS:
        # Word-boundary match against the abbreviation (preserve case
        # variations like 'mri' or 'MRI').
        pat = re.compile(rf"\b{re.escape(abbr)}\b", re.I)
        if pat.search(out) and full.lower() not in out.lower():
            # Append the long form once; keep the short form in place
            # (clinicians use them as synonyms in citations).
            out = pat.sub(f"{abbr} ({full})", out, count=1)
    return out


def _extract_subject_count(text: str, hints: dict[str, object]) -> None:
    for pat in (_RE_AT_LEAST_N_SUBJECTS, _RE_AT_LEAST_N_BARE):
        m = pat.search(text)
        if m:
            try:
                hints["min_subjects"] = int(m.group(1))
                return
            except ValueError:
                continue


def _extract_age(text: str, hints: dict[str, object]) -> None:
    m = _RE_AGE_BETWEEN.search(text)
    if m:
        try:
            lo, hi = int(m.group(1)), int(m.group(2))
            if 0 < lo < hi <= 120:
                hints["age_min"], hints["age_max"] = lo, hi
                return
        except ValueError:
            pass
    m = _RE_OLDER_THAN.search(text)
    if m:
        try:
            hints["age_min"] = int(m.group(1))
            return
        except ValueError:
            pass
    m = _RE_AGE_PLUS.search(text)
    if m:
        try:
            v = int(m.group(1))
            # Plausible age range only.
            if not (0 < v < 120):
                return
            # "100+ patients" is a subject count, not an age. If the
            # matched span is immediately followed by a subject/
            # patient/etc. word, drop it.
            tail = text[m.end():m.end() + 24].lstrip().lower()
            if tail.startswith((
                "subject", "patient", "participant", "case", "cohort", "sample",
            )):
                return
            hints["age_min"] = v
        except ValueError:
            pass
