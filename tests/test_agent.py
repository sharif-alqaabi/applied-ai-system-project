"""
Tests for the MusicAgent agentic workflow.

The API-calling run() method is not tested here because it requires a live
ANTHROPIC_API_KEY. Instead, the tests cover the rule-based helper methods
that the agent uses internally — the parts that do not need the network.
"""

import pytest
from src.agent import MusicAgent


@pytest.fixture
def agent():
    return MusicAgent()


@pytest.fixture
def chill_prefs():
    return {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.35,
        "danceability": 0.55,
        "likes_acoustic": True,
        "tempo_bpm": 76,
        "valence": 0.60,
    }


@pytest.fixture
def pop_prefs():
    return {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "danceability": 0.8,
        "likes_acoustic": False,
        "tempo_bpm": 122,
        "valence": 0.82,
    }


# ------------------------------------------------------------------
# evaluate_fit tests
# ------------------------------------------------------------------

def test_evaluate_fit_strong_match(agent, chill_prefs):
    agent._user_prefs = chill_prefs
    result = agent._evaluate_fit({
        "song_title": "Library Rain",
        "song_genre": "lofi",
        "song_mood": "chill",
        "song_energy": 0.35,
    })
    assert "strong fit" in result
    assert "genre is an exact match" in result
    assert "mood is an exact match" in result


def test_evaluate_fit_weak_match(agent, pop_prefs):
    agent._user_prefs = pop_prefs
    result = agent._evaluate_fit({
        "song_title": "Quiet Constellations",
        "song_genre": "classical",
        "song_mood": "peaceful",
        "song_energy": 0.18,
    })
    assert "weak fit" in result
    assert "genre mismatch" in result
    assert "mood mismatch" in result


def test_evaluate_fit_moderate_match(agent, pop_prefs):
    agent._user_prefs = pop_prefs
    result = agent._evaluate_fit({
        "song_title": "Rooftop Lights",
        "song_genre": "indie pop",
        "song_mood": "happy",
        "song_energy": 0.76,
    })
    assert "mood is an exact match" in result
    assert "genre mismatch" in result


def test_evaluate_fit_returns_advice_keep(agent, chill_prefs):
    agent._user_prefs = chill_prefs
    result = agent._evaluate_fit({
        "song_title": "Focus Flow",
        "song_genre": "lofi",
        "song_mood": "chill",
        "song_energy": 0.40,
    })
    assert "Keep this result" in result


def test_evaluate_fit_returns_advice_retry(agent, pop_prefs):
    agent._user_prefs = pop_prefs
    result = agent._evaluate_fit({
        "song_title": "Quiet Constellations",
        "song_genre": "classical",
        "song_mood": "peaceful",
        "song_energy": 0.18,
    })
    assert "different scoring mode" in result


# ------------------------------------------------------------------
# _format_recs tests
# ------------------------------------------------------------------

def test_format_recs_empty(agent):
    result = agent._format_recs([])
    assert "No recommendations found" in result


def test_format_recs_shows_title_and_score(agent):
    recs = [
        (
            {
                "title": "Test Song",
                "artist": "Test Artist",
                "genre": "lofi",
                "mood": "chill",
                "energy": 0.35,
            },
            5.42,
            "genre match (+2.0), mood match (+1.5)",
        )
    ]
    result = agent._format_recs(recs)
    assert "Test Song" in result
    assert "5.42" in result
    assert "genre match" in result


# ------------------------------------------------------------------
# _build_user_message tests
# ------------------------------------------------------------------

def test_build_user_message_contains_profile_fields(agent, chill_prefs):
    msg = agent._build_user_message(chill_prefs, k=5)
    assert "lofi" in msg
    assert "chill" in msg
    assert "0.35" in msg


def test_build_user_message_k_is_included(agent, chill_prefs):
    msg = agent._build_user_message(chill_prefs, k=3)
    assert "3" in msg


# ------------------------------------------------------------------
# _best_cached_recs tests
# ------------------------------------------------------------------

def test_best_cached_recs_empty(agent):
    agent._recs_cache = {}
    recs, mode = agent._best_cached_recs()
    assert recs == []
    assert mode == "none"


def test_best_cached_recs_returns_last_mode(agent):
    fake_rec = [({}, 1.0, "reason")]
    agent._recs_cache = {"balanced": fake_rec, "genre-first": fake_rec}
    recs, mode = agent._best_cached_recs()
    assert mode == "genre-first"
    assert recs == fake_rec
