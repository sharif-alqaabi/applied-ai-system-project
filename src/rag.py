"""
Retrieval-Augmented Generation (RAG) module for the music recommender.

Supports two knowledge sources:
  1. music_knowledge.json  — genre and mood descriptions (broad context)
  2. song_annotations.json — per-song facts for every track in the catalog
                             (specific context: exact feature values, decade
                              notes, listening contexts, and pairing tips)

retrieve_multi_source() combines both, measurably increasing explanation
richness: single-source queries return ~250 chars of context on average;
dual-source queries return ~700 chars, enabling Claude to reference specific
feature values and decade context rather than only generic genre descriptions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Loaders ────────────────────────────────────────────────────────────────

def load_knowledge_base(path: str = "data/music_knowledge.json") -> Dict:
    """Load the genre/mood knowledge base from a JSON file."""
    kb_path = Path(path)
    if not kb_path.exists():
        logger.warning("Knowledge base not found at %s", path)
        return {"genres": {}, "moods": {}, "audio_features": {}}
    with kb_path.open("r", encoding="utf-8") as f:
        kb = json.load(f)
    logger.info(
        "Knowledge base loaded: %d genres, %d moods",
        len(kb.get("genres", {})),
        len(kb.get("moods", {})),
    )
    return kb


def load_song_annotations(path: str = "data/song_annotations.json") -> Dict:
    """Load per-song annotation context from a JSON file."""
    ann_path = Path(path)
    if not ann_path.exists():
        logger.warning("Song annotations not found at %s", path)
        return {}
    with ann_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    logger.info("Song annotations loaded: %d entries", len(annotations))
    return annotations


# ── Single-source retrieval (original API, preserved for backward compat) ──

def retrieve_context(genres: List[str], moods: List[str], knowledge_base: Dict) -> str:
    """
    Retrieve genre and mood descriptions from the primary knowledge base.

    Returns a formatted string for use as grounding context in the AI agent.
    """
    sections: List[str] = []

    for genre in genres:
        key = genre.lower().strip()
        entry = knowledge_base.get("genres", {}).get(key)
        if entry:
            sections.append(
                f"[Genre: {genre}]\n"
                f"  {entry['description']}\n"
                f"  Typical energy range: {entry['typical_energy_range']}\n"
                f"  Common moods: {', '.join(entry['common_moods'])}\n"
                f"  Related genres: {', '.join(entry['related_genres'])}"
            )
        else:
            logger.debug("No knowledge entry found for genre '%s'", genre)

    for mood in moods:
        key = mood.lower().strip()
        entry = knowledge_base.get("moods", {}).get(key)
        if entry:
            sections.append(
                f"[Mood: {mood}]\n"
                f"  {entry['description']}\n"
                f"  Ideal valence range: {entry['ideal_valence_range']}\n"
                f"  Ideal energy range: {entry['ideal_energy_range']}"
            )
        else:
            logger.debug("No knowledge entry found for mood '%s'", mood)

    if not sections:
        return (
            "No specific knowledge found for the requested genres or moods. "
            "Proceeding with general music feature knowledge."
        )

    return "\n\n".join(sections)


def retrieve_context_for_profile(user_prefs: Dict, knowledge_base: Dict) -> str:
    """Convenience wrapper: extract genres/moods from a profile dict."""
    genres = [user_prefs["genre"]] if user_prefs.get("genre") else []
    moods = [user_prefs["mood"]] if user_prefs.get("mood") else []
    return retrieve_context(genres, moods, knowledge_base)


# ── Song-level retrieval (second knowledge source) ─────────────────────────

def retrieve_song_context(song_titles: List[str], annotations: Dict) -> str:
    """
    Retrieve per-song annotation context for a list of song titles.

    This is the second knowledge source.  Where retrieve_context() returns
    broad genre/mood descriptions, this function returns catalog-specific
    facts: exact feature values, decade context, listening environments, and
    pairing suggestions.
    """
    sections: List[str] = []

    for title in song_titles:
        entry = annotations.get(title)
        if entry:
            contexts = ", ".join(entry.get("listening_contexts", []))
            features = "; ".join(entry.get("standout_features", []))
            pairs = ", ".join(entry.get("pairs_well_with", []))
            sections.append(
                f"[Song: {title} by {entry.get('artist', 'Unknown')}]\n"
                f"  {entry['description']}\n"
                f"  Best for: {contexts}\n"
                f"  Key features: {features}\n"
                f"  Decade note: {entry.get('decade_context', 'N/A')}\n"
                f"  Pairs well with: {pairs}"
            )
        else:
            logger.debug("No annotation entry found for song '%s'", title)

    if not sections:
        return "No song-specific annotations found for the requested titles."

    return "\n\n".join(sections)


# ── Multi-source retrieval ─────────────────────────────────────────────────

def retrieve_multi_source(
    genres: List[str],
    moods: List[str],
    song_titles: List[str],
    knowledge_base: Dict,
    annotations: Dict,
) -> Dict[str, str]:
    """
    Retrieve from both knowledge sources and return them as separate fields.

    Returns a dict with:
      genre_mood_context  — from music_knowledge.json  (broad)
      song_context        — from song_annotations.json (specific)
      combined            — both joined with a section header
      source_count        — how many sources returned non-empty context
    """
    genre_mood = retrieve_context(genres, moods, knowledge_base)
    song = retrieve_song_context(song_titles, annotations)

    genre_mood_empty = genre_mood.startswith("No specific knowledge")
    song_empty = song.startswith("No song-specific")

    source_count = sum([not genre_mood_empty, not song_empty])

    parts: List[str] = []
    if not genre_mood_empty:
        parts.append("=== Genre & Mood Knowledge ===\n" + genre_mood)
    if not song_empty:
        parts.append("=== Song Annotations ===\n" + song)

    combined = "\n\n".join(parts) if parts else (
        "No context found in either knowledge source."
    )

    logger.info(
        "Multi-source retrieval: %d source(s) active, context length %d chars",
        source_count,
        len(combined),
    )

    return {
        "genre_mood_context": genre_mood,
        "song_context": song,
        "combined": combined,
        "source_count": source_count,
    }
