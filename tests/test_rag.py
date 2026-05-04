"""Tests for the RAG knowledge retrieval module."""

import pytest
from src.rag import load_knowledge_base, retrieve_context, retrieve_context_for_profile


@pytest.fixture
def kb():
    return load_knowledge_base("data/music_knowledge.json")


def test_knowledge_base_loads(kb):
    assert "genres" in kb
    assert "moods" in kb
    assert "audio_features" in kb


def test_knowledge_base_has_expected_genres(kb):
    for genre in ["pop", "lofi", "rock", "jazz", "classical"]:
        assert genre in kb["genres"], f"Expected genre '{genre}' in knowledge base"


def test_knowledge_base_has_expected_moods(kb):
    for mood in ["happy", "chill", "intense", "focused", "moody"]:
        assert mood in kb["moods"], f"Expected mood '{mood}' in knowledge base"


def test_genre_entry_has_required_fields(kb):
    entry = kb["genres"]["lofi"]
    assert "description" in entry
    assert "typical_energy_range" in entry
    assert "common_moods" in entry
    assert "related_genres" in entry


def test_mood_entry_has_required_fields(kb):
    entry = kb["moods"]["chill"]
    assert "description" in entry
    assert "ideal_valence_range" in entry
    assert "ideal_energy_range" in entry


def test_retrieve_context_genre(kb):
    result = retrieve_context(["lofi"], [], kb)
    assert "lofi" in result.lower()
    assert "energy" in result.lower()


def test_retrieve_context_mood(kb):
    result = retrieve_context([], ["chill"], kb)
    assert "chill" in result.lower()
    assert "valence" in result.lower()


def test_retrieve_context_both(kb):
    result = retrieve_context(["pop"], ["happy"], kb)
    assert "[Genre: pop]" in result
    assert "[Mood: happy]" in result


def test_retrieve_context_unknown_returns_fallback(kb):
    result = retrieve_context(["unknown_genre_xyz"], ["unknown_mood_xyz"], kb)
    assert "No specific knowledge found" in result


def test_retrieve_context_case_insensitive(kb):
    result = retrieve_context(["LoFi"], ["CHILL"], kb)
    assert "lofi" in result.lower()
    assert "chill" in result.lower()


def test_retrieve_context_for_profile(kb):
    profile = {"genre": "rock", "mood": "intense"}
    result = retrieve_context_for_profile(profile, kb)
    assert "rock" in result.lower()
    assert "intense" in result.lower()


def test_retrieve_context_for_profile_missing_fields(kb):
    result = retrieve_context_for_profile({}, kb)
    assert isinstance(result, str)


def test_load_knowledge_base_missing_file():
    kb = load_knowledge_base("data/nonexistent_file.json")
    assert kb == {"genres": {}, "moods": {}, "audio_features": {}}
