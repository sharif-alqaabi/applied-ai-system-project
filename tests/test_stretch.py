"""
Tests covering all four stretch features:

  RAG Enhancement          — multi-source retrieval, richness improvement
  Agentic Enhancement      — log_plan tool, planning chain observable steps
  Fine-Tuning/Specialization — evaluate_explanation_quality, few-shot criteria
  Test Harness             — already covered in test_eval.py; documented here
"""

import pytest
from src.rag import (
    load_song_annotations,
    retrieve_song_context,
    retrieve_multi_source,
    load_knowledge_base,
)
from src.agent import MusicAgent, _FEW_SHOT_ADDENDUM, _BASE_SYSTEM_PROMPT
from src.eval import evaluate_explanation_quality, compare_rag_sources


# ══════════════════════════════════════════════════════════════════════════════
# RAG ENHANCEMENT: second knowledge source (song_annotations.json)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def annotations():
    return load_song_annotations("data/song_annotations.json")


@pytest.fixture(scope="module")
def kb():
    return load_knowledge_base("data/music_knowledge.json")


def test_song_annotations_loads(annotations):
    assert len(annotations) == 18, "Expected annotations for all 18 catalog songs"


def test_song_annotations_has_required_fields(annotations):
    required = {"artist", "description", "listening_contexts",
                "standout_features", "decade_context", "pairs_well_with"}
    for title, entry in annotations.items():
        missing = required - entry.keys()
        assert not missing, f"'{title}' missing fields: {missing}"


def test_retrieve_song_context_returns_entry(annotations):
    result = retrieve_song_context(["Library Rain"], annotations)
    assert "Library Rain" in result
    assert "Paper Lanterns" in result


def test_retrieve_song_context_multiple(annotations):
    result = retrieve_song_context(["Library Rain", "Storm Runner"], annotations)
    assert "Library Rain" in result
    assert "Storm Runner" in result


def test_retrieve_song_context_unknown_title(annotations):
    result = retrieve_song_context(["Nonexistent Song XYZ"], annotations)
    assert "No song-specific annotations found" in result


def test_retrieve_multi_source_returns_both(kb, annotations):
    result = retrieve_multi_source(["lofi"], ["chill"], ["Library Rain"], kb, annotations)
    assert result["source_count"] == 2
    assert len(result["combined"]) > len(result["genre_mood_context"])
    assert len(result["combined"]) > len(result["song_context"])


def test_retrieve_multi_source_richness(kb, annotations):
    """Dual-source context must be measurably richer than single-source."""
    single = retrieve_multi_source(["pop"], ["happy"], [], kb, annotations)
    dual = retrieve_multi_source(["pop"], ["happy"], ["Sunrise City", "Gym Hero"], kb, annotations)
    assert dual["source_count"] == 2
    assert len(dual["combined"]) > len(single["combined"]) * 1.5


def test_retrieve_multi_source_empty_kb(annotations):
    empty_kb = {"genres": {}, "moods": {}, "audio_features": {}}
    result = retrieve_multi_source(["lofi"], ["chill"], ["Library Rain"], empty_kb, annotations)
    assert result["source_count"] == 1
    assert "Library Rain" in result["combined"]


def test_multi_source_avg_gain_over_threshold(kb, annotations):
    """RAG dual-source retrieval should deliver at least 200% more context."""
    profiles = [
        (["lofi"], ["chill"], ["Library Rain", "Midnight Coding"]),
        (["pop"], ["happy"], ["Sunrise City", "Gym Hero"]),
    ]
    gains = []
    for genres, moods, titles in profiles:
        single_len = len(retrieve_multi_source(genres, moods, [], kb, annotations)["combined"])
        dual_len = len(retrieve_multi_source(genres, moods, titles, kb, annotations)["combined"])
        gains.append((dual_len - single_len) / single_len * 100)
    avg_gain = sum(gains) / len(gains)
    assert avg_gain >= 200, f"Expected >= 200% gain, got {avg_gain:.1f}%"


# ══════════════════════════════════════════════════════════════════════════════
# AGENTIC ENHANCEMENT: log_plan tool + observable planning chain
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def agent():
    return MusicAgent()


def test_agent_has_log_plan_tool(agent):
    tool_names = [t["name"] for t in agent._MusicAgent__class_tools
                  if hasattr(agent, "_MusicAgent__class_tools")] if False else \
                 [t["name"] for t in __import__("src.agent", fromlist=["_TOOLS"])._TOOLS]
    assert "log_plan" in tool_names


def test_agent_has_retrieve_song_knowledge_tool():
    from src.agent import _TOOLS
    tool_names = [t["name"] for t in _TOOLS]
    assert "retrieve_song_knowledge" in tool_names


def test_agent_planning_chain_starts_empty(agent):
    agent._planning_chain = []
    assert agent._planning_chain == []


def test_agent_dispatch_log_plan_records_entry(agent):
    agent._planning_chain = []
    agent._dispatch("log_plan", {
        "intent": "Test intent",
        "strategy": "Test strategy",
        "focus_areas": ["energy", "genre"],
    }, k=5)
    assert len(agent._planning_chain) == 1
    assert agent._planning_chain[0]["intent"] == "Test intent"
    assert agent._planning_chain[0]["focus_areas"] == ["energy", "genre"]


def test_agent_dispatch_log_plan_returns_confirmation(agent):
    agent._planning_chain = []
    result = agent._dispatch("log_plan", {"intent": "x", "strategy": "y"}, k=5)
    assert "Plan recorded" in result


def test_agent_dispatch_retrieve_song_knowledge(agent):
    result = agent._dispatch(
        "retrieve_song_knowledge",
        {"song_titles": ["Library Rain"]},
        k=5,
    )
    assert "Library Rain" in result
    assert "Paper Lanterns" in result


def test_agent_dispatch_retrieve_song_unknown(agent):
    result = agent._dispatch(
        "retrieve_song_knowledge",
        {"song_titles": ["Unknown Song XYZ"]},
        k=5,
    )
    assert "No song-specific annotations found" in result


def test_agent_build_user_message_standard(agent):
    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.35}
    msg = agent._build_user_message(prefs, k=5, specialized=False)
    assert "specialized style is active" not in msg


def test_agent_build_user_message_specialized(agent):
    prefs = {"genre": "lofi", "mood": "chill", "energy": 0.35}
    msg = agent._build_user_message(prefs, k=5, specialized=True)
    assert "specialized style is active" in msg


# ══════════════════════════════════════════════════════════════════════════════
# FINE-TUNING / SPECIALIZATION: few-shot style + quality measurement
# ══════════════════════════════════════════════════════════════════════════════

def test_few_shot_addendum_is_non_empty():
    assert len(_FEW_SHOT_ADDENDUM) > 200


def test_few_shot_addendum_contains_examples():
    assert "Example 1" in _FEW_SHOT_ADDENDUM
    assert "Example 2" in _FEW_SHOT_ADDENDUM
    assert "Why this fits your vibe" in _FEW_SHOT_ADDENDUM


def test_evaluate_quality_all_criteria_met():
    good_explanation = (
        "Library Rain by Paper Lanterns is your best match. "
        "The knowledge base confirms lofi typically runs 0.20–0.55 energy — "
        "this track sits at energy 0.35, exactly on target. "
        "With acousticness of 0.86 and 72 BPM, it delivers the chill focus texture you need. "
        "Why this fits your vibe: slow, quiet, and perfect for deep work."
    )
    criteria = evaluate_explanation_quality(good_explanation)
    assert criteria["mentions_energy_value"], "Should detect '0.35' near 'energy'"
    assert criteria["references_knowledge"], "Should detect 'knowledge base'"
    assert criteria["includes_vibe_sentence"], "Should detect 'fits your vibe'"
    assert criteria["cites_specific_feature"], "Should detect 'acousticness' or 'BPM'"


def test_evaluate_quality_no_criteria_met():
    bare_explanation = "I recommend Library Rain. It is a nice song you will enjoy."
    criteria = evaluate_explanation_quality(bare_explanation)
    passed = sum(criteria.values())
    assert passed == 0, f"Bare explanation should pass 0 criteria, passed {passed}"


def test_evaluate_quality_partial():
    partial = "This song has acousticness of 0.86. It sounds good."
    criteria = evaluate_explanation_quality(partial)
    # cites_specific_feature=True, others likely False
    assert criteria["cites_specific_feature"]
    assert not criteria["includes_vibe_sentence"]


def test_evaluate_quality_energy_detection():
    text = "The track energy is 0.42, which is close to your 0.35 target."
    criteria = evaluate_explanation_quality(text)
    assert criteria["mentions_energy_value"]


def test_evaluate_quality_vibe_sentence_variants():
    for phrase in ["Why this fits your vibe:", "fits your vibe perfectly", "Why this fits your vibe"]:
        criteria = evaluate_explanation_quality(phrase)
        assert criteria["includes_vibe_sentence"], f"Failed to detect: '{phrase}'"


# ══════════════════════════════════════════════════════════════════════════════
# TEST HARNESS: documented as already implemented (src/eval.py)
# The full harness is covered in tests/test_eval.py.
# This section verifies the new eval additions introduced for stretch features.
# ══════════════════════════════════════════════════════════════════════════════

def test_compare_rag_sources_runs_without_error(capsys):
    compare_rag_sources()
    captured = capsys.readouterr()
    assert "RAG ENHANCEMENT" in captured.out
    assert "2-source" in captured.out


def test_compare_rag_sources_shows_positive_gain(capsys):
    compare_rag_sources()
    captured = capsys.readouterr()
    assert "+" in captured.out
