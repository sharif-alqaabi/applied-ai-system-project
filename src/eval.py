"""
Reliability evaluation harness for the Applied AI Music Recommender.

Covers all four stretch features:

  TEST HARNESS (required)
    run_benchmark() — 6 predefined cases, confidence scores, pass/fail report

  RAG ENHANCEMENT
    compare_rag_sources() — measures context richness for 1-source vs. 2-source
    retrieval, showing the measurable improvement from adding song_annotations.json

  FINE-TUNING / SPECIALIZATION
    evaluate_explanation_quality() — scores an AI explanation on 4 measurable
    style criteria; specialized mode should score 3–4, standard 0–2

  AGENTIC WORKFLOW ENHANCEMENT
    The planning chain is observable via agent.run()['planning_chain']; the
    benchmark and comparison functions log every intermediate step at INFO level

Run from the repo root:
    python -m src.eval

Exit code: 0 if all cases pass, 1 if any fail (CI-friendly).
"""

import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.recommender import compute_confidence, load_songs, recommend_songs_by_mode
from src.rag import (
    load_knowledge_base,
    load_song_annotations,
    retrieve_context,
    retrieve_multi_source,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Confidence thresholds ──────────────────────────────────────────────────
_STRONG = 0.90
_MODERATE = 0.70


def _confidence_label(c: float) -> str:
    if c >= _STRONG:
        return "strong fit"
    if c >= _MODERATE:
        return "moderate fit"
    return "weak fit"


# ── Benchmark definition ───────────────────────────────────────────────────

@dataclass
class TestCase:
    id: str
    name: str
    profile: Dict
    expected_song: str   # exact title that must appear in top_k results
    top_k: int = 3
    note: str = ""


BENCHMARK: List[TestCase] = [
    TestCase(
        id="TC-01",
        name="Chill Lofi — genre + mood + energy all aligned",
        profile={
            "genre": "lofi", "mood": "chill", "energy": 0.35,
            "danceability": 0.55, "likes_acoustic": True,
            "tempo_bpm": 76, "valence": 0.60,
        },
        expected_song="Library Rain",
        top_k=1,
    ),
    TestCase(
        id="TC-02",
        name="High-Energy Pop — exact genre + mood match",
        profile={
            "genre": "pop", "mood": "happy", "energy": 0.80,
            "danceability": 0.80, "likes_acoustic": False,
            "tempo_bpm": 122, "valence": 0.82,
        },
        expected_song="Sunrise City",
        top_k=1,
    ),
    TestCase(
        id="TC-03",
        name="Deep Intense Rock — high energy + heavy genre",
        profile={
            "genre": "rock", "mood": "intense", "energy": 0.92,
            "danceability": 0.50, "likes_acoustic": False,
            "tempo_bpm": 150, "valence": 0.45,
        },
        expected_song="Storm Runner",
        top_k=1,
    ),
    TestCase(
        id="TC-04",
        name="EDM Euphoria — maximum energy, danceability, and valence",
        profile={
            "genre": "edm", "mood": "excited", "energy": 0.95,
            "danceability": 0.91, "likes_acoustic": False,
            "tempo_bpm": 128, "valence": 0.74,
        },
        expected_song="Neon Sprint",
        top_k=1,
    ),
    TestCase(
        id="TC-05",
        name="Folk Nostalgic — acoustic, low-energy, no exact mood song",
        profile={
            "genre": "folk", "mood": "nostalgic", "energy": 0.31,
            "danceability": 0.44, "likes_acoustic": True,
            "tempo_bpm": 84, "valence": 0.68,
        },
        expected_song="Porchlight Letters",
        top_k=2,
        note=(
            "Catalog has no folk+nostalgic song; Porchlight Letters (folk+calm) "
            "wins on genre, energy, and acoustic fit but misses the mood match."
        ),
    ),
    TestCase(
        id="TC-06",
        name="Conflicted Edge Case — classical genre vs. high energy target",
        profile={
            "genre": "classical", "mood": "moody", "energy": 0.92,
            "danceability": 0.25, "likes_acoustic": True,
            "tempo_bpm": 65, "valence": 0.30,
        },
        expected_song="Quiet Constellations",
        top_k=2,
        note=(
            "Known limitation: the 2.0-point genre bonus overrides a 0.74 "
            "energy mismatch, producing a low confidence score. "
            "The recommender picks the right genre but ignores the energy conflict."
        ),
    ),
]


# ── Result dataclass ───────────────────────────────────────────────────────

@dataclass
class EvalResult:
    case: TestCase
    passed: bool
    rank: Optional[int]           # actual rank of expected_song (1-indexed), None if not found
    top_score: float              # score of the #1 result
    confidence: float             # normalized 0.0–1.0 confidence for the #1 result
    top_song_title: str


# ── Core evaluation logic ──────────────────────────────────────────────────

def run_case(case: TestCase, songs: List[Dict]) -> EvalResult:
    """Run one benchmark test case and return a structured result."""
    logger.info("[%s] Running: %s", case.id, case.name)

    recs = recommend_songs_by_mode(case.profile, songs, mode_name="balanced", k=case.top_k)

    if not recs:
        logger.error("[%s] Recommender returned no results", case.id)
        return EvalResult(
            case=case, passed=False, rank=None,
            top_score=0.0, confidence=0.0, top_song_title="(none)",
        )

    top_song, top_score, _ = recs[0]
    top_title = top_song["title"]

    # Check whether the expected song appears within the required rank
    found_rank: Optional[int] = None
    for i, (song, _, _) in enumerate(recs, start=1):
        if song["title"] == case.expected_song:
            found_rank = i
            break

    passed = found_rank is not None

    # Confidence is computed for the #1 result (not the expected song)
    confidence = compute_confidence(top_score, case.profile)

    status = "PASS" if passed else "FAIL"
    rank_str = f"rank {found_rank}" if found_rank else "not found"
    logger.info(
        "[%s] %s — '%s' %s in top-%d  |  top score %.2f  |  confidence %.3f (%s)",
        case.id, status, case.expected_song, rank_str, case.top_k,
        top_score, confidence, _confidence_label(confidence),
    )
    if case.note:
        logger.info("[%s] Note: %s", case.id, case.note)

    return EvalResult(
        case=case,
        passed=passed,
        rank=found_rank,
        top_score=top_score,
        confidence=confidence,
        top_song_title=top_title,
    )


def run_benchmark(songs_path: str = "data/songs.csv") -> List[EvalResult]:
    """Load songs and run all benchmark test cases."""
    songs = load_songs(songs_path)
    logger.info("Loaded %d songs from %s", len(songs), songs_path)
    return [run_case(case, songs) for case in BENCHMARK]


# ── Report formatting ──────────────────────────────────────────────────────

def print_report(results: List[EvalResult]) -> None:
    """Print a human-readable reliability report to stdout."""
    W = 70
    passed_count = sum(1 for r in results if r.passed)
    total = len(results)
    pass_rate = passed_count / total * 100 if total else 0.0

    confidences = [r.confidence for r in results]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    min_result = min(results, key=lambda r: r.confidence)
    max_result = max(results, key=lambda r: r.confidence)

    print()
    print("=" * W)
    print("  RELIABILITY EVALUATION — Applied AI Music Recommender")
    print("=" * W)
    print(f"  Benchmark : {total} test cases against data/songs.csv")
    print(f"  Mode      : balanced (default scoring)")
    print("-" * W)

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        rank_str = f"rank {r.rank}" if r.rank else "NOT FOUND"
        conf_label = _confidence_label(r.confidence)
        print()
        print(
            f"  {r.case.id}  {status}  "
            f"'{r.case.expected_song}' at {rank_str} "
            f"(expected in top-{r.case.top_k})"
        )
        print(f"         Profile    : {r.case.profile.get('genre')}, "
              f"{r.case.profile.get('mood')}, "
              f"energy {r.case.profile.get('energy'):.2f}")
        print(f"         Top result : '{r.top_song_title}'  score {r.top_score:.2f}")
        print(f"         Confidence : {r.confidence:.3f}  ({conf_label})")
        if r.case.note:
            # Wrap note at 60 chars
            note = r.case.note
            print(f"         Note       : {note[:60]}")
            if len(note) > 60:
                print(f"                      {note[60:].lstrip()}")

    print()
    print("-" * W)
    print(f"  RESULTS    : {passed_count} / {total} passed  ({pass_rate:.1f}%)")
    print(
        f"  Confidence : avg {avg_conf:.3f}  |  "
        f"min {min_result.confidence:.3f} ({min_result.case.id})  |  "
        f"max {max_result.confidence:.3f} ({max_result.case.id})"
    )
    print("-" * W)

    # Narrative summary
    strong = sum(1 for r in results if r.confidence >= _STRONG)
    moderate = sum(1 for r in results if _MODERATE <= r.confidence < _STRONG)
    weak = sum(1 for r in results if r.confidence < _MODERATE)
    print(f"  Confidence breakdown:")
    print(f"    Strong  (>= {_STRONG}) : {strong} / {total}")
    print(f"    Moderate({_MODERATE}–{_STRONG}): {moderate} / {total}")
    print(f"    Weak    (<  {_MODERATE}) : {weak} / {total}")

    if any(not r.passed for r in results):
        failed = [r.case.id for r in results if not r.passed]
        print(f"  Failed cases : {', '.join(failed)}")

    print("=" * W)
    print()


# ── RAG richness comparison ────────────────────────────────────────────────

def compare_rag_sources(
    kb_path: str = "data/music_knowledge.json",
    ann_path: str = "data/song_annotations.json",
) -> None:
    """
    Measure and print the context richness difference between 1-source and
    2-source RAG retrieval.

    Single-source: genre + mood descriptions only (music_knowledge.json).
    Dual-source  : genre + mood + song annotations (adds song_annotations.json).

    Richness proxy: character count of retrieved context.  More context gives
    Claude more specific facts to cite, measurably improving explanation quality.
    """
    W = 70
    kb = load_knowledge_base(kb_path)
    annotations = load_song_annotations(ann_path)

    test_cases = [
        {
            "label": "Chill Lofi",
            "genres": ["lofi"], "moods": ["chill"],
            "song_titles": ["Library Rain", "Midnight Coding"],
        },
        {
            "label": "High-Energy Pop",
            "genres": ["pop"], "moods": ["happy"],
            "song_titles": ["Sunrise City", "Gym Hero"],
        },
        {
            "label": "Folk Nostalgic",
            "genres": ["folk"], "moods": ["nostalgic"],
            "song_titles": ["Porchlight Letters", "Golden Hour Drive"],
        },
    ]

    print()
    print("=" * W)
    print("  RAG ENHANCEMENT — Multi-Source Context Richness Comparison")
    print("=" * W)
    print(f"  {'Profile':<20} {'1-source (chars)':>18} {'2-source (chars)':>18} {'Gain':>10}")
    print("-" * W)

    total_single, total_dual = 0, 0
    for tc in test_cases:
        single = retrieve_context(tc["genres"], tc["moods"], kb)
        dual_result = retrieve_multi_source(
            tc["genres"], tc["moods"], tc["song_titles"], kb, annotations
        )
        dual = dual_result["combined"]

        single_len = len(single)
        dual_len = len(dual)
        gain_pct = (dual_len - single_len) / single_len * 100 if single_len else 0.0

        total_single += single_len
        total_dual += dual_len

        logger.info(
            "RAG comparison [%s]: 1-source=%d chars, 2-source=%d chars (+%.0f%%)",
            tc["label"], single_len, dual_len, gain_pct,
        )
        print(
            f"  {tc['label']:<20} {single_len:>18,} {dual_len:>18,} "
            f"{gain_pct:>+9.0f}%"
        )

    avg_gain = (total_dual - total_single) / total_single * 100 if total_single else 0.0
    print("-" * W)
    print(
        f"  {'AVERAGE':<20} {total_single // len(test_cases):>18,} "
        f"{total_dual // len(test_cases):>18,} {avg_gain:>+9.0f}%"
    )
    print(f"  Adding song_annotations.json provides {avg_gain:.0f}% more context on average.")
    print(f"  This enables Claude to cite specific feature values and decade notes")
    print(f"  rather than only generic genre/mood descriptions.")
    print("=" * W)
    print()


# ── Explanation quality measurement ───────────────────────────────────────

def evaluate_explanation_quality(text: str) -> Dict[str, bool]:
    """
    Score an AI explanation against 4 measurable style criteria.

    Specialized mode (few-shot) should pass all 4.
    Standard mode typically passes 0–2.

    Criteria
    --------
    mentions_energy_value  : explanation cites a numeric energy figure (e.g. 0.35)
    references_knowledge   : explanation references the retrieved genre/mood context
    includes_vibe_sentence : explanation ends with a 'Why this fits your vibe:' line
    cites_specific_feature : explanation names at least one specific audio feature
                             (acousticness, BPM, danceability, valence, etc.)
    """
    tl = text.lower()
    return {
        "mentions_energy_value": bool(
            re.search(r'\benergy\b[^\n]{0,40}0\.\d+', tl)
            or re.search(r'0\.\d+[^\n]{0,30}\benergy\b', tl)
        ),
        "references_knowledge": any(
            kw in tl for kw in [
                "retrieved", "knowledge", "typically", "range", "confirms",
                "knowledge base", "genre knowledge", "annotation",
            ]
        ),
        "includes_vibe_sentence": (
            "why this fits" in tl or "fits your vibe" in tl
        ),
        "cites_specific_feature": any(
            kw in tl for kw in [
                "acousticness", "bpm", "danceability", "valence",
                "instrumentalness", "tempo", "liveliness",
            ]
        ),
    }


def print_quality_report(label: str, text: str) -> int:
    """Print quality criteria check and return the number of criteria met."""
    criteria = evaluate_explanation_quality(text)
    passed = sum(criteria.values())
    print(f"\n  Quality check — {label}  ({passed}/4 criteria)")
    for name, result in criteria.items():
        mark = "PASS" if result else "FAIL"
        print(f"    [{mark}] {name.replace('_', ' ')}")
    return passed


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> int:
    results = run_benchmark()
    print_report(results)

    compare_rag_sources()

    all_passed = all(r.passed for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
