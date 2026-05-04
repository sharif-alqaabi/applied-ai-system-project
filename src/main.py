"""
Command line runner for the Music Recommender Simulation.

Usage:
  python -m src.main          # run all scoring-based profiles and experiments
  python -m src.main --ai     # run the Claude-powered agentic recommender

The --ai mode activates two advanced features:
  - RAG: genre/mood knowledge is retrieved from data/music_knowledge.json
    before any recommendation is generated, grounding Claude's reasoning
    in factual audio-feature descriptions.
  - Agentic workflow: Claude plans (retrieves knowledge), acts (runs the
    recommender), checks its own work (evaluates fit), and optionally retries
    with a different scoring mode before writing its final answer.

Requires ANTHROPIC_API_KEY to be set in the environment for --ai mode.
"""

import logging
import os
import sys

from src.recommender import (
    DEFAULT_WEIGHTS,
    SCORING_MODES,
    load_songs,
    recommend_songs,
    recommend_songs_by_mode,
    recommend_songs_with_config,
)

PROFILE_LIBRARY = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "danceability": 0.8,
        "likes_acoustic": False,
        "tempo_bpm": 122,
        "valence": 0.82,
    },
    "Chill Lofi": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.35,
        "danceability": 0.55,
        "likes_acoustic": True,
        "tempo_bpm": 76,
        "valence": 0.60,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.92,
        "danceability": 0.50,
        "likes_acoustic": False,
        "tempo_bpm": 150,
        "valence": 0.45,
    },
    "Conflicted Edge Case": {
        "genre": "classical",
        "mood": "moody",
        "energy": 0.92,
        "danceability": 0.25,
        "likes_acoustic": True,
        "tempo_bpm": 65,
        "valence": 0.30,
    },
}

EXPERIMENTAL_WEIGHTS = {
    **DEFAULT_WEIGHTS,
    "genre": 1.0,
    "energy": 3.0,
}

PROFILE_MODE_DEMOS = [
    ("High-Energy Pop", "genre-first"),
    ("Chill Lofi", "mood-first"),
    ("Deep Intense Rock", "energy-focused"),
]


def print_recommendations(title: str, recommendations: list[tuple[dict, float, str]]) -> None:
    """Print a readable block of recommendations for one profile."""
    print(f"\n=== {title} ===\n")
    for index, rec in enumerate(recommendations, start=1):
        song, score, explanation = rec
        print(f"{index}. {song['title']} by {song['artist']}")
        print(f"   Score: {score:.2f}")
        print(f"   Reasons: {explanation}")
        print(f"   Vibe: {song['genre']}, {song['mood']}, energy {song['energy']:.2f}")
        print()


def run_profile(name: str, user_prefs: dict, songs: list[dict]) -> None:
    """Run and print one recommendation profile using the default logic."""
    recommendations = recommend_songs(user_prefs, songs, k=5)
    print_recommendations(name, recommendations)


def run_experiment(profile_name: str, user_prefs: dict, songs: list[dict]) -> None:
    """Compare baseline recommendations with an energy-heavy experiment."""
    baseline = recommend_songs(user_prefs, songs, k=5)
    experiment = recommend_songs_with_config(
        user_prefs,
        songs,
        k=5,
        weights=EXPERIMENTAL_WEIGHTS,
    )

    print("\n=== Weight Shift Experiment ===")
    print("Baseline weights: genre 2.0, mood 1.5, energy 1.5")
    print("Experimental weights: genre 1.0, mood 1.5, energy 3.0")
    print(f"Profile tested: {profile_name}")

    print_recommendations("Baseline Top 5", baseline)
    print_recommendations("Experimental Top 5", experiment)


def run_mode_demo(profile_name: str, mode_name: str, user_prefs: dict, songs: list[dict]) -> None:
    """Show how a named scoring mode changes the ranking strategy."""
    mode = SCORING_MODES[mode_name]
    recommendations = recommend_songs_by_mode(user_prefs, songs, mode_name=mode_name, k=5)
    print(f"\n=== {mode_name.title()} Mode ===")
    print(f"Profile tested: {profile_name}")
    print(f"Strategy: {mode['description']}")
    print_recommendations(f"{profile_name} ({mode_name})", recommendations)


def run_ai_mode() -> None:
    """
    Run the Claude-powered agentic recommender for two demo profiles.

    The agent:
      1. Retrieves genre/mood knowledge from the RAG knowledge base
      2. Runs the scoring recommender to get candidates
      3. Evaluates whether the top result fits the listener
      4. Retries with a different scoring mode if fit is weak
      5. Writes a natural-language summary referencing retrieved context
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "\n[ERROR] ANTHROPIC_API_KEY is not set.\n"
            "Export it before running --ai mode:\n"
            "  export ANTHROPIC_API_KEY=your-key-here\n"
        )
        sys.exit(1)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from src.agent import MusicAgent

    agent = MusicAgent()

    demo_profiles = ["Chill Lofi", "High-Energy Pop"]

    for profile_name in demo_profiles:
        prefs = PROFILE_LIBRARY[profile_name]
        print(f"\n{'='*60}")
        print(f"  AI-Enhanced Recommendations — {profile_name}")
        print(f"{'='*60}")
        print(f"  Genre: {prefs['genre']}  |  Mood: {prefs['mood']}  |  Energy: {prefs['energy']}")
        print()

        result = agent.run(prefs, k=5)

        print(result["explanation"])
        print(
            f"\n[Agent used {result['iterations']} iteration(s), "
            f"final scoring mode: {result['mode_used']}]"
        )

        if result["recommendations"]:
            print("\nFinal ranked songs:")
            for i, (song, score, reasons) in enumerate(result["recommendations"], start=1):
                print(f"  {i}. {song['title']} by {song['artist']}  (score: {score:.2f})")


def main() -> None:
    if "--ai" in sys.argv:
        run_ai_mode()
        return

    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    for name, user_prefs in PROFILE_LIBRARY.items():
        run_profile(name, user_prefs, songs)

    run_experiment("High-Energy Pop", PROFILE_LIBRARY["High-Energy Pop"], songs)

    for profile_name, mode_name in PROFILE_MODE_DEMOS:
        run_mode_demo(profile_name, mode_name, PROFILE_LIBRARY[profile_name], songs)


if __name__ == "__main__":
    main()
