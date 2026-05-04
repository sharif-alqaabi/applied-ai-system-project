"""
Retrieval-Augmented Generation (RAG) module for the music recommender.

Loads a curated knowledge base of genre and mood descriptions, then retrieves
relevant context for a given user profile. The retrieved context is passed to
the AI agent to enrich its reasoning before generating recommendations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def load_knowledge_base(path: str = "data/music_knowledge.json") -> Dict:
    """Load the music knowledge base from a JSON file."""
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


def retrieve_context(genres: List[str], moods: List[str], knowledge_base: Dict) -> str:
    """
    Retrieve relevant knowledge for a list of genre and mood names.

    Returns a formatted string that the AI agent uses as grounding context
    before it decides which songs to recommend and how to explain them.
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
    """Convenience wrapper that extracts genres and moods from a user profile dict."""
    genres = [user_prefs["genre"]] if user_prefs.get("genre") else []
    moods = [user_prefs["mood"]] if user_prefs.get("mood") else []
    return retrieve_context(genres, moods, knowledge_base)
