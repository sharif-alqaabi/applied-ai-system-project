"""
Agentic music recommender powered by the Claude API.

Stretch features implemented here:

  AGENTIC WORKFLOW ENHANCEMENT
    A log_plan tool forces Claude to declare its intent, tool strategy, and
    focus areas before acting.  Every plan entry is stored in _planning_chain
    and returned in the result dict, making the full reasoning chain
    observable in the CLI output.

  RAG ENHANCEMENT
    A retrieve_song_knowledge tool gives Claude access to the second knowledge
    source (song_annotations.json), enabling it to cite specific feature
    values, decade context, and listening environments for individual tracks.

  FINE-TUNING / SPECIALIZATION
    Passing specialized=True to run() appends two few-shot examples to the
    system prompt that constrain explanation style: cite exact energy values,
    reference retrieved knowledge, and end every recommendation with a
    'Why this fits your vibe:' sentence.  evaluate_explanation_quality() in
    eval.py measures the difference.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from src.rag import (
    load_knowledge_base,
    load_song_annotations,
    retrieve_context,
    retrieve_song_context,
)
from src.recommender import load_songs, recommend_songs_by_mode

logger = logging.getLogger(__name__)

# ── System prompts ─────────────────────────────────────────────────────────

_BASE_SYSTEM_PROMPT = """\
You are an expert music recommendation assistant with deep knowledge of
music genres, moods, and audio features.

Your job is to help listeners discover songs that truly match their taste.
You have four tools available:

  log_plan               — declare your intent and tool strategy FIRST
  retrieve_music_knowledge — look up genre/mood descriptions (broad context)
  retrieve_song_knowledge  — look up per-song facts for specific titles
  get_recommendations    — run the scoring engine and get ranked candidates
  evaluate_fit           — assess whether the top result fits the listener

Workflow you MUST follow:
1. Call log_plan to declare what you will do and why.
2. Call retrieve_music_knowledge for the listener's genre and mood.
3. Call get_recommendations with mode "balanced".
4. Call retrieve_song_knowledge for the top 2–3 results to get specific facts.
5. Call evaluate_fit on the top result.
6. If fit is weak (score < 4), call get_recommendations with a better mode,
   then re-evaluate.
7. Write a recommendation summary that cites specific feature values from what
   you retrieved — do not invent facts.\
"""

_FEW_SHOT_ADDENDUM = """

=== SPECIALIZATION EXAMPLES ===

Below are two examples of the explanation style required in specialized mode.
Match this style exactly: cite the exact energy value, reference retrieved
knowledge, and end every top recommendation with a "Why this fits your vibe:"
sentence.

--- Example 1 ---
Listener: lofi, chill, energy 0.35, acoustic

Top recommendation: Library Rain by Paper Lanterns

Library Rain is your strongest match. The genre/mood knowledge confirms that
lofi typically operates between 0.20–0.55 energy — Library Rain sits at
exactly 0.35, landing dead-center in that range. Its acousticness of 0.86
pairs directly with your acoustic preference, and the song annotation notes
its 72 BPM tempo creates the unhurried pace that defines the chill mood.
Instrumentalness of 0.88 means near-zero vocal content, so nothing will
break your focus.

Why this fits your vibe: This is the sonic equivalent of a productive rainy
afternoon — slow, textured, and quiet enough to think in.

--- Example 2 ---
Listener: rock, intense, energy 0.92, not acoustic

Top recommendation: Storm Runner by Voltline

Storm Runner hits every target. The knowledge base notes rock typically runs
0.60–1.0 energy; Storm Runner at 0.91 matches your 0.92 target within 0.01.
The song annotation confirms 152 BPM and minimal acousticness (0.10) —
pure electric production with no acoustic softening. The aggressive mood
alignment means this track was built for exactly the intensity you want.

Why this fits your vibe: Storm Runner is for the moments when you need music
to push you — loud, driving, and unrelenting.

=== END EXAMPLES ===
Always use this style. Every top recommendation must include exact numeric
values from the retrieved context and end with "Why this fits your vibe:".
"""

# ── Tool definitions ───────────────────────────────────────────────────────

_TOOLS = [
    {
        "name": "log_plan",
        "description": (
            "Declare your reasoning plan BEFORE taking any action. "
            "Call this first every time so your decision chain is observable."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "description": "What you are trying to accomplish for this listener.",
                },
                "strategy": {
                    "type": "string",
                    "description": "Which tools you plan to call and in what order.",
                },
                "focus_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific profile aspects you will prioritize (e.g. energy, genre match).",
                },
            },
            "required": ["intent", "strategy"],
        },
    },
    {
        "name": "retrieve_music_knowledge",
        "description": (
            "Look up broad genre/mood descriptions from the primary knowledge base. "
            "Use this to understand the typical energy range, common moods, and "
            "related genres before recommending."
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
        "name": "retrieve_song_knowledge",
        "description": (
            "Look up per-song annotation facts for specific tracks — exact feature "
            "values, decade context, listening environments, and pairing suggestions. "
            "Call this after get_recommendations to get specific facts about the top results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "song_titles": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Exact song titles to retrieve annotations for.",
                },
            },
            "required": ["song_titles"],
        },
    },
    {
        "name": "get_recommendations",
        "description": (
            "Run the scoring engine and return ranked song candidates "
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
                        "if evaluate_fit returns a weak result."
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
            "Returns a numeric fit score (0–7) and plain-language advice."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "song_title": {"type": "string"},
                "song_genre": {"type": "string"},
                "song_mood": {"type": "string"},
                "song_energy": {"type": "number"},
            },
            "required": ["song_title", "song_genre", "song_mood", "song_energy"],
        },
    },
]


# ── MusicAgent ─────────────────────────────────────────────────────────────

class MusicAgent:
    """
    Agentic music recommender with observable planning chain and
    optional few-shot specialization.
    """

    MAX_ITERATIONS = 10

    def __init__(
        self,
        songs_path: str = "data/songs.csv",
        knowledge_path: str = "data/music_knowledge.json",
        annotations_path: str = "data/song_annotations.json",
    ) -> None:
        self.songs = load_songs(songs_path)
        self.knowledge_base = load_knowledge_base(knowledge_path)
        self.annotations = load_song_annotations(annotations_path)
        self.client = anthropic.Anthropic()
        self._user_prefs: Dict = {}
        self._recs_cache: Dict[str, List[Tuple]] = {}
        self._planning_chain: List[Dict] = []
        logger.info(
            "MusicAgent ready — %d songs, %d annotations loaded",
            len(self.songs),
            len(self.annotations),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        user_prefs: Dict,
        k: int = 5,
        specialized: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full agentic recommendation loop.

        Parameters
        ----------
        user_prefs  : listener profile dict
        k           : number of songs to recommend
        specialized : if True, append few-shot examples to the system prompt
                      to constrain explanation style (measurably different output)

        Returns
        -------
        dict with keys:
          explanation    — natural-language recommendation text
          recommendations — list of (song_dict, score, reason_str) tuples
          iterations     — number of agent turns used
          mode_used      — final scoring mode
          planning_chain — list of plan dicts logged by log_plan calls
          specialized    — whether few-shot mode was active
        """
        self._user_prefs = user_prefs
        self._recs_cache = {}
        self._planning_chain = []

        system_text = _BASE_SYSTEM_PROMPT
        if specialized:
            system_text += _FEW_SHOT_ADDENDUM

        messages: List[Dict] = [
            {"role": "user", "content": self._build_user_message(user_prefs, k, specialized)}
        ]

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            logger.info("Agent iteration %d (specialized=%s)", iteration, specialized)

            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=[
                    {
                        "type": "text",
                        "text": system_text,
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
                logger.info(
                    "Agent finished in %d iterations, mode=%s, plans=%d",
                    iteration, mode_used, len(self._planning_chain),
                )
                return {
                    "explanation": final_text,
                    "recommendations": best_recs,
                    "iterations": iteration,
                    "mode_used": mode_used,
                    "planning_chain": list(self._planning_chain),
                    "specialized": specialized,
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
            "planning_chain": list(self._planning_chain),
            "specialized": specialized,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_user_message(self, user_prefs: Dict, k: int, specialized: bool) -> str:
        style_note = (
            "  Note: specialized style is active — cite exact numeric values "
            "and end each top recommendation with 'Why this fits your vibe:'.\n"
            if specialized else ""
        )
        return (
            f"Please recommend {k} songs for a listener with these preferences:\n\n"
            f"  Favorite genre  : {user_prefs.get('genre', 'not specified')}\n"
            f"  Favorite mood   : {user_prefs.get('mood', 'not specified')}\n"
            f"  Energy target   : {user_prefs.get('energy', 'not specified')} "
            f"(0 = very calm, 1 = very intense)\n"
            f"  Likes acoustic  : {user_prefs.get('likes_acoustic', 'not specified')}\n"
            f"  Danceability    : {user_prefs.get('danceability', 'not specified')}\n"
            f"{style_note}\n"
            "Follow your workflow: log_plan first, then retrieve knowledge, "
            "get candidates, look up song annotations, evaluate fit, and write your answer."
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
        if name == "log_plan":
            entry = {
                "intent": inputs.get("intent", ""),
                "strategy": inputs.get("strategy", ""),
                "focus_areas": inputs.get("focus_areas", []),
            }
            self._planning_chain.append(entry)
            logger.info(
                "Plan recorded — intent: %s | focus: %s",
                entry["intent"][:80],
                entry["focus_areas"],
            )
            return "Plan recorded. Proceed with your strategy."

        if name == "retrieve_music_knowledge":
            genres = inputs.get("genres", [])
            moods = inputs.get("moods", [])
            context = retrieve_context(genres, moods, self.knowledge_base)
            logger.info("RAG source 1: genre/mood context for %s / %s", genres, moods)
            return context

        if name == "retrieve_song_knowledge":
            titles = inputs.get("song_titles", [])
            context = retrieve_song_context(titles, self.annotations)
            logger.info("RAG source 2: song annotations for %s", titles)
            return context

        if name == "get_recommendations":
            mode = inputs.get("mode", "balanced")
            num = inputs.get("k", k)
            recs = recommend_songs_by_mode(
                self._user_prefs, self.songs, mode_name=mode, k=num
            )
            self._recs_cache[mode] = recs
            logger.info("Recommender: %d results via mode='%s'", len(recs), mode)
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
        if not self._recs_cache:
            return [], "none"
        mode = list(self._recs_cache.keys())[-1]
        return self._recs_cache[mode], mode
