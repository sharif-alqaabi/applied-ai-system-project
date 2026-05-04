import csv
from typing import List, Dict, Tuple
from dataclasses import dataclass

DEFAULT_WEIGHTS = {
    "genre": 2.0,
    "mood": 1.5,
    "energy": 1.5,
    "danceability": 0.75,
    "acousticness": 0.5,
    "tempo_bpm": 0.25,
    "valence": 0.25,
    "popularity": 0.35,
    "release_decade": 0.5,
    "detailed_mood_tag": 1.0,
    "instrumentalness": 0.35,
    "liveliness": 0.35,
    "lyric_density": 0.35,
}

SCORING_MODES = {
    "balanced": {
        "description": "Default mode that balances genre, mood, and audio similarity.",
        "weights": DEFAULT_WEIGHTS,
        "use_mood": True,
    },
    "genre-first": {
        "description": "Prioritizes exact genre matches before other signals.",
        "weights": {
            **DEFAULT_WEIGHTS,
            "genre": 3.5,
            "mood": 1.0,
            "energy": 1.2,
            "detailed_mood_tag": 0.8,
        },
        "use_mood": True,
    },
    "mood-first": {
        "description": "Prioritizes emotional fit and detailed mood over genre.",
        "weights": {
            **DEFAULT_WEIGHTS,
            "genre": 1.0,
            "mood": 2.75,
            "valence": 0.5,
            "detailed_mood_tag": 1.5,
        },
        "use_mood": True,
    },
    "energy-focused": {
        "description": "Pushes high similarity on energy and nearby audio intensity.",
        "weights": {
            **DEFAULT_WEIGHTS,
            "genre": 1.0,
            "mood": 1.0,
            "energy": 3.0,
            "danceability": 1.0,
            "liveliness": 0.6,
            "tempo_bpm": 0.4,
        },
        "use_mood": True,
    },
}

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    popularity: int = 0
    release_decade: int = 0
    detailed_mood_tag: str = ""
    instrumentalness: float = 0.0
    liveliness: float = 0.0
    lyric_density: float = 0.0

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    preferred_popularity: int | None = None
    preferred_release_decade: int | None = None
    preferred_detailed_mood_tag: str | None = None
    target_instrumentalness: float | None = None
    target_liveliness: float | None = None
    target_lyric_density: float | None = None

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return the top-k songs ranked by fit with the user profile."""
        scored_songs = sorted(
            self.songs,
            key=lambda song: self._score_song_object(user, song),
            reverse=True,
        )
        return scored_songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Describe why a song matched the user profile."""
        _, reasons = self._score_song_object(user, song, return_reasons=True)
        return ", ".join(reasons)

    def _score_song_object(
        self,
        user: UserProfile,
        song: Song,
        return_reasons: bool = False,
    ) -> float | Tuple[float, List[str]]:
        """Score a Song dataclass using the same logic as the functional API."""
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
            "preferred_popularity": user.preferred_popularity,
            "preferred_release_decade": user.preferred_release_decade,
            "preferred_detailed_mood_tag": user.preferred_detailed_mood_tag,
            "target_instrumentalness": user.target_instrumentalness,
            "target_liveliness": user.target_liveliness,
            "target_lyric_density": user.target_lyric_density,
        }
        song_dict = {
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
            "tempo_bpm": song.tempo_bpm,
            "valence": song.valence,
            "danceability": song.danceability,
            "acousticness": song.acousticness,
            "popularity": song.popularity,
            "release_decade": song.release_decade,
            "detailed_mood_tag": song.detailed_mood_tag,
            "instrumentalness": song.instrumentalness,
            "liveliness": song.liveliness,
            "lyric_density": song.lyric_density,
        }
        score, reasons = score_song(user_prefs, song_dict)
        if return_reasons:
            return score, reasons
        return score


def compute_confidence(
    score: float,
    user_prefs: Dict,
    weights: Dict[str, float] | None = None,
) -> float:
    """
    Return a normalized 0.0–1.0 confidence value for a raw recommendation score.

    Confidence = score / theoretical_max, where theoretical_max is the highest
    possible score achievable given the features present in user_prefs and the
    active weights.  A score of 1.0 means the top song matched every specified
    preference perfectly; 0.0 means no features matched at all.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    max_score = 0.0

    if user_prefs.get("genre") is not None:
        max_score += w["genre"]
    if user_prefs.get("mood") is not None:
        max_score += w["mood"]
    if user_prefs.get("energy") is not None:
        max_score += w["energy"]
    # Danceability falls back to energy when not explicitly set (mirrors scoring logic)
    if user_prefs.get("danceability") is not None or user_prefs.get("energy") is not None:
        max_score += w["danceability"]
    if user_prefs.get("likes_acoustic") is not None:
        max_score += w["acousticness"]
    if user_prefs.get("tempo_bpm") is not None:
        max_score += w["tempo_bpm"]
    if user_prefs.get("valence") is not None:
        max_score += w["valence"]
    if _first_pref(user_prefs, "preferred_popularity", "popularity_target") is not None:
        max_score += w["popularity"]
    if _first_pref(user_prefs, "preferred_release_decade", "release_decade") is not None:
        max_score += w["release_decade"]
    if _first_pref(user_prefs, "preferred_detailed_mood_tag", "detailed_mood_tag") is not None:
        max_score += w["detailed_mood_tag"]
    if _first_pref(user_prefs, "target_instrumentalness", "instrumentalness") is not None:
        max_score += w["instrumentalness"]
    if _first_pref(user_prefs, "target_liveliness", "liveliness") is not None:
        max_score += w["liveliness"]
    if _first_pref(user_prefs, "target_lyric_density", "lyric_density") is not None:
        max_score += w["lyric_density"]

    if max_score <= 0.0:
        return 0.0
    return round(min(1.0, score / max_score), 3)


def _first_pref(user_prefs: Dict, *keys: str):
    """Return the first non-None preference value from a list of possible keys."""
    for key in keys:
        value = user_prefs.get(key)
        if value is not None:
            return value
    return None


def get_scoring_mode(mode_name: str = "balanced") -> Dict:
    """Return a named scoring mode configuration."""
    mode_key = mode_name.strip().lower()
    return SCORING_MODES.get(mode_key, SCORING_MODES["balanced"])


def apply_diversity_penalty(
    song: Dict,
    base_score: float,
    selected_songs: List[Dict],
    artist_penalty: float = 0.9,
    genre_penalty: float = 0.45,
) -> Tuple[float, List[str]]:
    """Reduce repeated artists or genres while building the final ranked list."""
    adjusted_score = base_score
    penalties: List[str] = []

    selected_artists = {selected_song["artist"] for selected_song in selected_songs}
    selected_genres = {selected_song["genre"] for selected_song in selected_songs}

    if song["artist"] in selected_artists:
        adjusted_score -= artist_penalty
        penalties.append(f"artist diversity penalty (-{artist_penalty:.2f})")

    if song["genre"] in selected_genres:
        adjusted_score -= genre_penalty
        penalties.append(f"genre diversity penalty (-{genre_penalty:.2f})")

    return round(adjusted_score, 2), penalties

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from CSV and convert numeric fields into math-friendly types."""
    songs: List[Dict] = []
    int_fields = {"id"}
    float_fields = {
        "energy",
        "tempo_bpm",
        "valence",
        "danceability",
        "acousticness",
        "instrumentalness",
        "liveliness",
        "lyric_density",
    }
    int_fields = int_fields | {"popularity", "release_decade"}

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            song: Dict = {}
            for key, value in row.items():
                if key in int_fields:
                    song[key] = int(value)
                elif key in float_fields:
                    song[key] = float(value)
                else:
                    song[key] = value
            songs.append(song)

    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score one song against user preferences and explain the main contributors."""
    return score_song_with_config(user_prefs, song)

def score_song_with_config(
    user_prefs: Dict,
    song: Dict,
    weights: Dict[str, float] | None = None,
    use_mood: bool = True,
) -> Tuple[float, List[str]]:
    """Score one song with configurable weights for evaluation experiments."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    score = 0.0
    reasons: List[str] = []

    if song["genre"] == user_prefs.get("genre"):
        genre_points = weights["genre"]
        score += genre_points
        reasons.append(f"genre match (+{genre_points:.2f})")

    if use_mood and song["mood"] == user_prefs.get("mood"):
        mood_points = weights["mood"]
        score += mood_points
        reasons.append(f"mood match (+{mood_points:.2f})")

    target_energy = user_prefs.get("energy")
    if target_energy is not None:
        energy_similarity = max(0.0, 1 - abs(song["energy"] - target_energy))
        energy_points = round(energy_similarity * weights["energy"], 2)
        score += energy_points
        reasons.append(f"energy similarity (+{energy_points:.2f})")

    target_danceability = user_prefs.get("danceability", target_energy)
    if target_danceability is not None:
        dance_similarity = max(0.0, 1 - abs(song["danceability"] - target_danceability))
        dance_points = round(dance_similarity * weights["danceability"], 2)
        score += dance_points
        reasons.append(f"danceability similarity (+{dance_points:.2f})")

    likes_acoustic = user_prefs.get("likes_acoustic")
    if likes_acoustic is not None:
        acoustic_match = song["acousticness"] if likes_acoustic else 1 - song["acousticness"]
        acoustic_points = round(acoustic_match * weights["acousticness"], 2)
        score += acoustic_points
        reasons.append(f"acoustic fit (+{acoustic_points:.2f})")

    target_tempo = user_prefs.get("tempo_bpm")
    if target_tempo is not None:
        tempo_similarity = max(0.0, 1 - (abs(song["tempo_bpm"] - target_tempo) / 100))
        tempo_points = round(tempo_similarity * weights["tempo_bpm"], 2)
        score += tempo_points
        reasons.append(f"tempo similarity (+{tempo_points:.2f})")

    target_valence = user_prefs.get("valence")
    if target_valence is not None:
        valence_similarity = max(0.0, 1 - abs(song["valence"] - target_valence))
        valence_points = round(valence_similarity * weights["valence"], 2)
        score += valence_points
        reasons.append(f"valence similarity (+{valence_points:.2f})")

    preferred_popularity = _first_pref(user_prefs, "preferred_popularity", "popularity_target", "popularity")
    if preferred_popularity is not None and "popularity" in song:
        popularity_similarity = max(
            0.0,
            1 - (abs(float(song["popularity"]) - float(preferred_popularity)) / 100),
        )
        popularity_points = round(popularity_similarity * weights["popularity"], 2)
        score += popularity_points
        reasons.append(f"popularity fit (+{popularity_points:.2f})")

    preferred_release_decade = _first_pref(
        user_prefs,
        "preferred_release_decade",
        "release_decade",
    )
    if preferred_release_decade is not None and "release_decade" in song:
        decade_gap = abs(int(song["release_decade"]) - int(preferred_release_decade))
        release_similarity = max(0.0, 1 - (decade_gap / 40))
        release_points = round(release_similarity * weights["release_decade"], 2)
        score += release_points
        reasons.append(f"release decade fit (+{release_points:.2f})")

    preferred_detailed_mood = _first_pref(
        user_prefs,
        "preferred_detailed_mood_tag",
        "detailed_mood_tag",
    )
    if preferred_detailed_mood is not None:
        song_detailed_mood = str(song.get("detailed_mood_tag", "")).strip().lower()
        preferred_detailed_mood = str(preferred_detailed_mood).strip().lower()
        if song_detailed_mood and song_detailed_mood == preferred_detailed_mood:
            detailed_mood_points = weights["detailed_mood_tag"]
            score += detailed_mood_points
            reasons.append(f"detailed mood match (+{detailed_mood_points:.2f})")

    target_instrumentalness = _first_pref(user_prefs, "target_instrumentalness", "instrumentalness")
    if target_instrumentalness is not None and "instrumentalness" in song:
        instrumental_similarity = max(0.0, 1 - abs(song["instrumentalness"] - target_instrumentalness))
        instrumental_points = round(instrumental_similarity * weights["instrumentalness"], 2)
        score += instrumental_points
        reasons.append(f"instrumentalness similarity (+{instrumental_points:.2f})")

    target_liveliness = _first_pref(user_prefs, "target_liveliness", "liveliness")
    if target_liveliness is not None and "liveliness" in song:
        liveliness_similarity = max(0.0, 1 - abs(song["liveliness"] - target_liveliness))
        liveliness_points = round(liveliness_similarity * weights["liveliness"], 2)
        score += liveliness_points
        reasons.append(f"liveliness similarity (+{liveliness_points:.2f})")

    target_lyric_density = _first_pref(user_prefs, "target_lyric_density", "lyric_density")
    if target_lyric_density is not None and "lyric_density" in song:
        lyric_similarity = max(0.0, 1 - abs(song["lyric_density"] - target_lyric_density))
        lyric_points = round(lyric_similarity * weights["lyric_density"], 2)
        score += lyric_points
        reasons.append(f"lyric density similarity (+{lyric_points:.2f})")

    return round(score, 2), reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Score, rank, and return the top-k songs with short explanations."""
    return recommend_songs_with_config(user_prefs, songs, k=k)

def recommend_songs_with_config(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    weights: Dict[str, float] | None = None,
    use_mood: bool = True,
) -> List[Tuple[Dict, float, str]]:
    """Rank songs with configurable scoring settings for experiments."""
    scored_songs: List[Tuple[Dict, float, List[str]]] = []

    for song in songs:
        score, reasons = score_song_with_config(
            user_prefs,
            song,
            weights=weights,
            use_mood=use_mood,
        )
        scored_songs.append((song, score, reasons))

    remaining_songs = sorted(scored_songs, key=lambda item: item[1], reverse=True)
    reranked_results: List[Tuple[Dict, float, str]] = []
    selected_song_dicts: List[Dict] = []

    while remaining_songs and len(reranked_results) < k:
        best_index = 0
        best_candidate: Tuple[Dict, float, List[str]] | None = None
        best_adjusted_score = float("-inf")
        best_reasons: List[str] = []

        for index, (song, base_score, reasons) in enumerate(remaining_songs):
            adjusted_score, penalty_reasons = apply_diversity_penalty(song, base_score, selected_song_dicts)
            if adjusted_score > best_adjusted_score:
                best_index = index
                best_candidate = (song, adjusted_score, reasons)
                best_adjusted_score = adjusted_score
                best_reasons = penalty_reasons

        assert best_candidate is not None
        song, adjusted_score, reasons = best_candidate
        explanation_parts = reasons + best_reasons if best_reasons else reasons
        reranked_results.append((song, adjusted_score, ", ".join(explanation_parts)))
        selected_song_dicts.append(song)
        remaining_songs.pop(best_index)

    return reranked_results


def recommend_songs_by_mode(
    user_prefs: Dict,
    songs: List[Dict],
    mode_name: str = "balanced",
    k: int = 5,
) -> List[Tuple[Dict, float, str]]:
    """Rank songs using one of the named scoring strategies."""
    mode = get_scoring_mode(mode_name)
    return recommend_songs_with_config(
        user_prefs,
        songs,
        k=k,
        weights=mode["weights"],
        use_mood=mode["use_mood"],
    )
