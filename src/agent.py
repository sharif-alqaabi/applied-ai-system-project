"""
Agentic music recommender powered by the Claude API.

Implements a plan-act-check loop:
  1. Plan  — retrieve genre/mood knowledge from the RAG knowledge base
  2. Act   — run the scoring-based recommender to get song candidates
  3. Check — evaluate whether the top result genuinely fits the user
  4. Refine — if fit is weak, retry with a different scoring mode

The retrieved context actively changes what Claude says and whether it
accepts the initial ranking or asks for a re-run, making RAG and the
agentic loop both fully integrated into the recommendation output.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import anthropic

from src.rag import load_knowledge_base, retrieve_context
from src.recommender import load_songs, recommend_songs_by_mode

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert music recommendation assistant with deep knowledge of \
music genres, moods, and audio features.

Your job is to help listeners discover songs that truly match their taste. \
You have three tools available:

  • retrieve_music_knowledge — look up rich descriptions of genres and moods \
from a curated knowledge base (RAG step)
  • get_recommendations — run the scoring engine and get ranked song candidates
  • evaluate_fit — assess whether the top result genuinely matches the \
listener's intent

Workflow you MUST follow:
1. Call retrieve_music_knowledge for the listener's genre and mood first.
2. Call get_recommendations with mode "balanced".
3. Call evaluate_fit on the top result.
4. If fit is weak (score < 4), call get_recommendations again with a better \
mode (genre-first, mood-first, or energy-focused), then re-evaluate.
5. Once satisfied, write a short, friendly recommendation summary that \
references specific audio features and explains why each song fits.

Be concrete. Reference energy levels, mood tags, and genre knowledge from \
what you retrieved. Do not just list scores.\
"""

_TOOLS = [
    {
        "name": "retrieve_music_knowledge",
        "description": (
            "Look up music knowledge about specific genres and moods from the "
            "curated knowledge base. Always call this first so you understand "
            "what the listener actually wants before recommending anything."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genres": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Genre names to retrieve knowledge about.",
                },
                "moods": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Mood names to retrieve knowledge about.",
                },
            },
            "required": ["genres", "moods"],
        },
    },
    {
        "name": "get_recommendations",
        "description": (
            "Run the music recommender engine and return ranked song candidates "
            "for the current listener profile."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["balanced", "genre-first", "mood-first", "energy-focused"],
                    "description": (
                        "Scoring strategy. Start with 'balanced'. Switch modes "
                        "if the top result evaluated as a weak fit."
                    ),
                },
                "k": {
                    "type": "integer",
                    "description": "Number of songs to return (default: 5).",
                    "default": 5,
                },
            },
            "required": ["mode"],
        },
    },
    {
        "name": "evaluate_fit",
        "description": (
            "Evaluate whether a specific song is a strong fit for the listener. "
            "Returns a numeric fit score and plain-language notes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "song_title": {"type": "string", "description": "Title of the song."},
                "song_genre": {"type": "string", "description": "Genre of the song."},
                "song_mood": {"type": "string", "description": "Mood tag of the song."},
                "song_energy": {
                    "type": "number",
                    "description": "Energy level of the song (0.0–1.0).",
                },
            },
            "required": ["song_title", "song_genre", "song_mood", "song_energy"],
        },
    },
]


class MusicAgent:
    """
    Agentic music recommender.

    Wraps the scoring-based recommender with a Claude-powered plan-act-check
    loop. The agent retrieves knowledge (RAG), scores candidates, evaluates
    fit, and may retry with a different scoring mode before returning a
    natural-language recommendation.
    """

    MAX_ITERATIONS = 8

    def __init__(
        self,
        songs_path: str = "data/songs.csv",
        knowledge_path: str = "data/music_knowledge.json",
    ) -> None:
        self.songs = load_songs(songs_path)
        self.knowledge_base = load_knowledge_base(knowledge_path)
        self.client = anthropic.Anthropic()
        self._user_prefs: Dict = {}
        self._recs_cache: Dict[str, List[Tuple]] = {}
        logger.info("MusicAgent ready — %d songs loaded", len(self.songs))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_prefs: Dict, k: int = 5) -> Dict[str, Any]:
        """
        Run the full agentic recommendation loop.

        Returns a dict with:
          explanation   — natural-language recommendation text from Claude
          recommendations — list of (song_dict, score, reason_str) tuples
          iterations    — how many agent turns were used
          mode_used     — which scoring mode produced the final result
        """
        self._user_prefs = user_prefs
        self._recs_cache = {}

        messages: List[Dict] = [
            {"role": "user", "content": self._build_user_message(user_prefs, k)}
        ]

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            logger.info("Agent iteration %d", iteration)

            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=_TOOLS,
                messages=messages,
            )

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                final_text = next(
                    (b.text for b in response.content if hasattr(b, "text")),
                    "No recommendation text generated.",
                )
                best_recs, mode_used = self._best_cached_recs()
                logger.info("Agent finished in %d iterations, mode=%s", iteration, mode_used)
                return {
                    "explanation": final_text,
                    "recommendations": best_recs,
                    "iterations": iteration,
                    "mode_used": mode_used,
                }

            if response.stop_reason == "tool_use":
                tool_results = self._handle_tool_calls(response.content, k)
                messages.append({"role": "user", "content": tool_results})

        logger.warning("Agent reached max iterations (%d)", self.MAX_ITERATIONS)
        best_recs, mode_used = self._best_cached_recs()
        return {
            "explanation": "Agent reached its iteration limit without completing.",
            "recommendations": best_recs,
            "iterations": self.MAX_ITERATIONS,
            "mode_used": mode_used,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, user_prefs: Dict, k: int) -> str:
        return (
            f"Please recommend {k} songs for a listener with these preferences:\n\n"
            f"  Favorite genre  : {user_prefs.get('genre', 'not specified')}\n"
            f"  Favorite mood   : {user_prefs.get('mood', 'not specified')}\n"
            f"  Energy target   : {user_prefs.get('energy', 'not specified')} "
            f"(0 = very calm, 1 = very intense)\n"
            f"  Likes acoustic  : {user_prefs.get('likes_acoustic', 'not specified')}\n"
            f"  Danceability    : {user_prefs.get('danceability', 'not specified')}\n\n"
            "Follow your workflow: retrieve knowledge first, then get candidates, "
            "then evaluate the top result before writing your final answer."
        )

    def _handle_tool_calls(self, content_blocks: List, k: int) -> List[Dict]:
        results = []
        for block in content_blocks:
            if getattr(block, "type", None) != "tool_use":
                continue
            output = self._dispatch(block.name, block.input, k)
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output}
            )
        return results

    def _dispatch(self, name: str, inputs: Dict, k: int) -> str:
        if name == "retrieve_music_knowledge":
            genres = inputs.get("genres", [])
            moods = inputs.get("moods", [])
            context = retrieve_context(genres, moods, self.knowledge_base)
            logger.info("RAG retrieved context for genres=%s moods=%s", genres, moods)
            return context

        if name == "get_recommendations":
            mode = inputs.get("mode", "balanced")
            num = inputs.get("k", k)
            recs = recommend_songs_by_mode(
                self._user_prefs, self.songs, mode_name=mode, k=num
            )
            self._recs_cache[mode] = recs
            logger.info("Recommender returned %d results using mode='%s'", len(recs), mode)
            return self._format_recs(recs)

        if name == "evaluate_fit":
            return self._evaluate_fit(inputs)

        logger.warning("Unknown tool called: %s", name)
        return f"Error: unknown tool '{name}'"

    def _format_recs(self, recs: List[Tuple]) -> str:
        if not recs:
            return "No recommendations found for this profile."
        lines = []
        for i, (song, score, explanation) in enumerate(recs, start=1):
            lines.append(
                f"{i}. {song['title']} by {song['artist']}\n"
                f"   Genre: {song['genre']} | Mood: {song['mood']} | "
                f"Energy: {song['energy']:.2f} | Score: {score:.2f}\n"
                f"   Scoring reasons: {explanation}"
            )
        return "\n\n".join(lines)

    def _evaluate_fit(self, inputs: Dict) -> str:
        """
        Rule-based fit check that the agent can call to decide whether to
        accept the current ranking or switch scoring mode.
        """
        user_genre = self._user_prefs.get("genre", "")
        user_mood = self._user_prefs.get("mood", "")
        user_energy = float(self._user_prefs.get("energy", 0.5))

        song_title = inputs.get("song_title", "Unknown")
        song_genre = inputs.get("song_genre", "")
        song_mood = inputs.get("song_mood", "")
        song_energy = float(inputs.get("song_energy", 0.5))

        fit_score = 0
        notes: List[str] = []

        if song_genre.lower() == user_genre.lower():
            fit_score += 3
            notes.append("genre is an exact match")
        else:
            notes.append(
                f"genre mismatch (listener wants '{user_genre}', song is '{song_genre}')"
            )

        if song_mood.lower() == user_mood.lower():
            fit_score += 2
            notes.append("mood is an exact match")
        else:
            notes.append(
                f"mood mismatch (listener wants '{user_mood}', song is '{song_mood}')"
            )

        energy_diff = abs(song_energy - user_energy)
        if energy_diff < 0.2:
            fit_score += 2
            notes.append(f"energy is very close (diff {energy_diff:.2f})")
        elif energy_diff < 0.4:
            fit_score += 1
            notes.append(f"energy is moderately close (diff {energy_diff:.2f})")
        else:
            notes.append(f"energy is far from target (diff {energy_diff:.2f})")

        verdict = (
            "strong fit" if fit_score >= 6
            else "moderate fit" if fit_score >= 4
            else "weak fit"
        )
        advice = (
            "Keep this result."
            if fit_score >= 4
            else "Consider retrying with a different scoring mode."
        )

        return (
            f"Fit evaluation for '{song_title}':\n"
            f"  Score: {fit_score}/7 ({verdict})\n"
            f"  Notes: {'; '.join(notes)}\n"
            f"  Advice: {advice}"
        )

    def _best_cached_recs(self) -> Tuple[List[Tuple], str]:
        """Return the cached recommendations from the most recently used mode."""
        if not self._recs_cache:
            return [], "none"
        mode = list(self._recs_cache.keys())[-1]
        return self._recs_cache[mode], mode
