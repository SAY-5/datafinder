from __future__ import annotations

from datafinder.normalize import normalize


def test_lowercase_collapse_whitespace_strip_punct():
    nq = normalize("  Find me   knee MRI datasets!  ")
    assert "  " not in nq.text
    assert not nq.text.endswith("!")
    assert "MRI" in nq.text


def test_abbreviation_expansion_adds_long_form():
    nq = normalize("any MRI of the knee?")
    assert "magnetic resonance imaging" in nq.text.lower()
    # Original abbreviation kept for citations.
    assert "MRI" in nq.text


def test_abbreviation_skipped_when_long_form_present():
    nq = normalize("show me magnetic resonance imaging cohorts")
    # Should not double-add the expansion.
    assert nq.text.lower().count("magnetic resonance imaging") == 1


def test_min_subjects_at_least():
    assert normalize("at least 100 subjects").hints == {"min_subjects": 100}


def test_min_subjects_bare_n_plus():
    assert normalize("100+ patients").hints == {"min_subjects": 100}


def test_age_range_between():
    h = normalize("subjects aged 40 to 70").hints
    assert h["age_min"] == 40 and h["age_max"] == 70


def test_age_older_than():
    assert normalize("older than 55 years").hints == {"age_min": 55}


def test_age_plus():
    assert normalize("age 40+").hints == {"age_min": 40}


def test_combines_constraints():
    h = normalize("MRI knee, at least 50 patients, age 60+").hints
    assert h["min_subjects"] == 50
    assert h["age_min"] == 60


def test_does_not_misextract_dataset_size_as_age():
    # "OAI 2K subjects" — 2 is not a plausible age, must NOT land as age_min.
    h = normalize("OAI 2K cohort knee MRI").hints
    assert "age_min" not in h
