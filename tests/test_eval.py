"""
Tests for the reliability evaluation harness and confidence scoring.

Covers:
  - compute_confidence boundary conditions and scaling
  - run_case pass/fail logic
  - run_benchmark completeness (all 6 cases, no crashes)
  - report generation (smoke test)
"""

import io
import sys
import pytest

from src.recommender import compute_confidence, load_songs, DEFAULT_WEIGHTS
from src.eval import (
    BENCHMARK,
    EvalResult,
    run_benchmark,
    run_case,
    print_report,
    _confidence_label,
    _STRONG,
    _MODERATE,
)


# ── compute_confidence ─────────────────────────────────────────────────────

def test_confidence_is_between_zero_and_one():
    profile = {"genre": "lofi", "mood": "chill", "energy": 0.35,
               "danceability": 0.55, "likes_acoustic": True,
               "tempo_bpm": 76, "valence": 0.60}
    c = compute_confidence(6.65, profile)
    assert 0.0 <= c <= 1.0


def test_confidence_zero_score_returns_zero():
    profile = {"genre": "lofi", "mood": "chill", "energy": 0.5}
    assert compute_confidence(0.0, profile) == 0.0


def test_confidence_empty_profile_returns_zero():
    assert compute_confidence(5.0, {}) == 0.0


def test_confidence_perfect_score_returns_one():
    profile = {"genre": "pop", "mood": "happy", "energy": 0.8,
               "danceability": 0.8, "likes_acoustic": False,
               "tempo_bpm": 122, "valence": 0.82}
    max_score = (
        DEFAULT_WEIGHTS["genre"]
        + DEFAULT_WEIGHTS["mood"]
        + DEFAULT_WEIGHTS["energy"]
        + DEFAULT_WEIGHTS["danceability"]
        + DEFAULT_WEIGHTS["acousticness"]
        + DEFAULT_WEIGHTS["tempo_bpm"]
        + DEFAULT_WEIGHTS["valence"]
    )
    c = compute_confidence(max_score, profile)
    assert c == 1.0


def test_confidence_above_max_clamps_to_one():
    profile = {"genre": "pop", "energy": 0.8}
    c = compute_confidence(9999.0, profile)
    assert c == 1.0


def test_confidence_scales_linearly():
    profile = {"genre": "rock", "mood": "intense", "energy": 0.9,
               "danceability": 0.5, "likes_acoustic": False,
               "tempo_bpm": 150, "valence": 0.45}
    c_high = compute_confidence(6.54, profile)
    c_low = compute_confidence(3.27, profile)
    assert c_high > c_low


def test_confidence_custom_weights():
    profile = {"genre": "lofi", "energy": 0.35}
    custom = {**DEFAULT_WEIGHTS, "genre": 5.0}
    c = compute_confidence(5.0, profile, weights=custom)
    assert 0.0 < c <= 1.0


# ── _confidence_label ─────────────────────────────────────────────────────

def test_label_strong():
    assert _confidence_label(_STRONG) == "strong fit"
    assert _confidence_label(1.0) == "strong fit"


def test_label_moderate():
    assert _confidence_label(_MODERATE) == "moderate fit"
    assert _confidence_label((_STRONG + _MODERATE) / 2) == "moderate fit"


def test_label_weak():
    assert _confidence_label(0.0) == "weak fit"
    assert _confidence_label(_MODERATE - 0.01) == "weak fit"


# ── BENCHMARK definition ───────────────────────────────────────────────────

def test_benchmark_has_six_cases():
    assert len(BENCHMARK) == 6


def test_benchmark_ids_are_unique():
    ids = [tc.id for tc in BENCHMARK]
    assert len(ids) == len(set(ids))


def test_benchmark_profiles_have_required_keys():
    required = {"genre", "mood", "energy"}
    for tc in BENCHMARK:
        missing = required - tc.profile.keys()
        assert not missing, f"{tc.id} profile is missing keys: {missing}"


def test_benchmark_top_k_is_positive():
    for tc in BENCHMARK:
        assert tc.top_k >= 1


# ── run_case ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def songs():
    return load_songs("data/songs.csv")


def test_run_case_lofi_passes(songs):
    tc = next(t for t in BENCHMARK if t.id == "TC-01")
    result = run_case(tc, songs)
    assert result.passed
    assert result.rank == 1
    assert result.confidence >= _STRONG


def test_run_case_conflicted_passes_but_low_confidence(songs):
    tc = next(t for t in BENCHMARK if t.id == "TC-06")
    result = run_case(tc, songs)
    assert result.passed
    assert result.confidence < _STRONG


def test_run_case_returns_eval_result(songs):
    tc = BENCHMARK[0]
    result = run_case(tc, songs)
    assert isinstance(result, EvalResult)
    assert isinstance(result.passed, bool)
    assert isinstance(result.confidence, float)
    assert isinstance(result.top_song_title, str)


def test_run_case_empty_catalog_fails():
    tc = BENCHMARK[0]
    result = run_case(tc, songs=[])
    assert not result.passed
    assert result.confidence == 0.0


# ── run_benchmark ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def benchmark_results():
    return run_benchmark()


def test_run_benchmark_returns_six_results(benchmark_results):
    assert len(benchmark_results) == 6


def test_run_benchmark_all_pass(benchmark_results):
    failed = [r.case.id for r in benchmark_results if not r.passed]
    assert not failed, f"Benchmark cases failed: {failed}"


def test_run_benchmark_confidence_range(benchmark_results):
    for r in benchmark_results:
        assert 0.0 <= r.confidence <= 1.0, (
            f"{r.case.id} confidence {r.confidence} is out of range"
        )


def test_run_benchmark_avg_confidence_above_threshold(benchmark_results):
    avg = sum(r.confidence for r in benchmark_results) / len(benchmark_results)
    assert avg >= 0.80, f"Average confidence {avg:.3f} is below acceptable threshold"


def test_run_benchmark_tc06_lowest_confidence(benchmark_results):
    min_r = min(benchmark_results, key=lambda r: r.confidence)
    assert min_r.case.id == "TC-06", (
        "Expected TC-06 (conflicted profile) to have the lowest confidence"
    )


# ── print_report (smoke test) ─────────────────────────────────────────────

def test_print_report_produces_output(benchmark_results):
    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured
    try:
        print_report(benchmark_results)
    finally:
        sys.stdout = old_stdout
    output = captured.getvalue()
    assert "RELIABILITY EVALUATION" in output
    assert "RESULTS" in output
    assert "6 / 6" in output
