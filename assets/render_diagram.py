"""
Renders the system architecture diagram for the Applied AI Music Recommender
and saves it to assets/system_architecture.png.

Run from the repo root:
    python3 assets/render_diagram.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

# ── Colour palette ─────────────────────────────────────────────────────────
C = {
    "bg":        "#0F1117",
    "panel":     "#1A1D27",
    "border":    "#2E3250",
    "input":     "#1E3A5F",
    "input_b":   "#4A90D9",
    "data":      "#1B3A2A",
    "data_b":    "#3DBE7A",
    "rag":       "#2D1F52",
    "rag_b":     "#9B59F5",
    "agent":     "#3D2010",
    "agent_b":   "#F59B20",
    "engine":    "#1E2F45",
    "engine_b":  "#4A90D9",
    "eval":      "#3D1020",
    "eval_b":    "#F55A5A",
    "output":    "#1B3A2A",
    "output_b":  "#3DBE7A",
    "test":      "#2A2A1A",
    "test_b":    "#D4C547",
    "arrow":     "#6B7DB3",
    "arrow_hl":  "#F59B20",
    "text_h":    "#FFFFFF",
    "text_s":    "#B0B8D0",
    "text_d":    "#7A8299",
    "loop_bg":   "#1E1528",
    "loop_b":    "#6B3FA0",
}

fig, ax = plt.subplots(figsize=(18, 11))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")


# ── Helper: rounded box ────────────────────────────────────────────────────
def box(ax, x, y, w, h, fc, ec, label, sublabel="", icon="", lw=1.8, radius=0.25):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3,
    )
    ax.add_patch(patch)
    cy = y + h / 2
    if icon:
        ax.text(x + 0.28, cy + (0.1 if sublabel else 0), icon,
                fontsize=13, ha="left", va="center", color=ec, zorder=4)
        tx = x + 0.62
    else:
        tx = x + w / 2
        ha = "center"
    ha = "left" if icon else "center"
    ax.text(tx, cy + (0.13 if sublabel else 0), label,
            fontsize=9.5, fontweight="bold", ha=ha, va="center",
            color=C["text_h"], zorder=4)
    if sublabel:
        ax.text(tx, cy - 0.18, sublabel,
                fontsize=7.5, ha=ha, va="center", color=C["text_s"], zorder=4)


# ── Helper: arrow ──────────────────────────────────────────────────────────
def arrow(ax, x1, y1, x2, y2, label="", color=None, style="->", lw=1.6):
    color = color or C["arrow"]
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=5,
    )
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.08, my + 0.08, label,
                fontsize=7, ha="center", va="center",
                color=color, zorder=6,
                bbox=dict(fc=C["bg"], ec="none", pad=1.5))


# ── Helper: dashed region ──────────────────────────────────────────────────
def region(ax, x, y, w, h, label, fc, ec, lw=1.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.35",
        linewidth=lw, edgecolor=ec, facecolor=fc,
        linestyle="--", zorder=1,
    )
    ax.add_patch(patch)
    ax.text(x + 0.18, y + h - 0.22, label,
            fontsize=8, ha="left", va="top",
            color=ec, style="italic", zorder=2)


# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(9, 10.6, "Applied AI Music Recommender — System Architecture",
        fontsize=14, fontweight="bold", ha="center", va="center",
        color=C["text_h"])
ax.text(9, 10.25,
        "RAG retrieval  ·  Agentic plan-act-check loop  ·  Scoring engine  ·  Human testing",
        fontsize=9, ha="center", va="center", color=C["text_s"])

# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND REGIONS
# ══════════════════════════════════════════════════════════════════════════════
region(ax, 0.25, 0.4,  3.5,  9.5, "INPUT", "#111520", C["input_b"])
region(ax, 4.0,  0.4,  9.8,  9.5, "AGENTIC LOOP  (src/agent.py · Claude API)", C["loop_bg"], C["loop_b"], lw=1.8)
region(ax, 14.2, 0.4,  3.5,  9.5, "OUTPUT", "#111520", C["output_b"])

# ══════════════════════════════════════════════════════════════════════════════
# INPUT COLUMN
# ══════════════════════════════════════════════════════════════════════════════
# User profile
box(ax, 0.5, 8.2, 3.0, 0.9, C["input"], C["input_b"],
    "User Profile", "genre · mood · energy · acoustic")

# songs.csv
box(ax, 0.5, 6.5, 3.0, 0.85, C["data"], C["data_b"],
    "songs.csv", "18 songs, 13 features each")

# Knowledge base
box(ax, 0.5, 4.8, 3.0, 0.85, C["rag"], C["rag_b"],
    "music_knowledge.json", "15 genres · 15 moods")

# Tests
box(ax, 0.5, 1.0, 3.0, 1.6, C["test"], C["test_b"],
    "Test Suite", "26 tests total\ntests/test_rag.py\ntests/test_agent.py\ntests/test_recommender.py", lw=1.4)

# ══════════════════════════════════════════════════════════════════════════════
# AGENTIC LOOP INTERNALS
# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — RAG Retriever
box(ax, 4.3, 8.0, 3.5, 1.1, C["rag"], C["rag_b"],
    "1  RAG Retriever", "src/rag.py  |  retrieve_context()", lw=2.2)

# Step 2 — Claude (planner / writer)
box(ax, 8.3, 6.5, 5.2, 2.8, C["agent"], C["agent_b"],
    "2  Claude  (claude-sonnet-4-6)", "MusicAgent.run()  |  tool_use loop\nplan  ->  act  ->  check  ->  refine  ->  explain",
    lw=2.5)

# Step 3 — Scoring Engine
box(ax, 4.3, 5.2, 3.5, 1.1, C["engine"], C["engine_b"],
    "3  Scoring Engine", "src/recommender.py\nrecommend_songs_by_mode()", lw=2.2)

# Step 4 — Evaluator
box(ax, 4.3, 3.5, 3.5, 1.1, C["eval"], C["eval_b"],
    "4  Fit Evaluator", "agent._evaluate_fit()\nscore 0-7 : strong / weak", lw=2.2)

# Scoring modes label
ax.text(4.55, 2.85, "Scoring modes:",
        fontsize=7.5, color=C["text_s"], va="top")
for i, mode in enumerate(["balanced", "genre-first", "mood-first", "energy-focused"]):
    ax.text(4.55, 2.55 - i * 0.26, f"• {mode}",
            fontsize=7.2, color=C["text_d"], va="top",
            fontfamily="monospace")

# Retry arrow (loop-back) — from evaluator back up to scoring engine
ax.annotate(
    "", xy=(5.85, 6.25), xytext=(5.85, 4.6),
    arrowprops=dict(
        arrowstyle="->", color=C["eval_b"], lw=1.6,
        connectionstyle="arc3,rad=-0.5",
    ),
    zorder=5,
)
ax.text(4.08, 5.42, "retry with\ndiff. mode",
        fontsize=7, ha="center", va="center", color=C["eval_b"],
        style="italic")

# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT COLUMN
# ══════════════════════════════════════════════════════════════════════════════
box(ax, 14.4, 7.8, 3.1, 1.3, C["output"], C["output_b"],
    "Ranked Songs", "top-k with scores\n+ diversity penalty")

box(ax, 14.4, 5.8, 3.1, 1.7, C["agent"], C["agent_b"],
    "AI Explanation", "natural language summary\nreferencing retrieved\ngenre/mood context")

box(ax, 14.4, 3.8, 3.1, 1.3, C["test"], C["test_b"],
    "Human Review", "verify tone, accuracy\nspot-check edge cases")

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS
# ══════════════════════════════════════════════════════════════════════════════
# User profile → RAG Retriever
arrow(ax, 3.5, 8.65, 4.3, 8.55, "user genre + mood", color=C["rag_b"])
# User profile → Claude
arrow(ax, 3.5, 8.65, 8.3, 7.8, "full profile", color=C["agent_b"])

# Knowledge base → RAG Retriever
arrow(ax, 3.5, 5.22, 4.3, 8.35, "JSON entries", color=C["rag_b"])

# RAG Retriever → Claude  (knowledge context)
arrow(ax, 7.8, 8.55, 8.3, 8.1, "retrieved context", color=C["rag_b"])

# Claude → Scoring Engine (tool call)
arrow(ax, 8.3, 7.5, 7.8, 5.75, "get_recommendations\n(tool call)", color=C["agent_b"])

# songs.csv → Scoring Engine
arrow(ax, 3.5, 6.93, 4.3, 5.77, "song catalog", color=C["data_b"])

# Scoring Engine → Claude (candidates)
arrow(ax, 7.8, 5.75, 8.3, 7.2, "ranked candidates", color=C["engine_b"])

# Claude → Evaluator (tool call)
arrow(ax, 8.3, 6.9, 7.8, 4.05, "evaluate_fit\n(tool call)", color=C["agent_b"])

# Evaluator → Claude (fit verdict)
arrow(ax, 7.8, 4.05, 8.3, 6.65, "fit score + advice", color=C["eval_b"])

# Claude → Ranked Songs
arrow(ax, 13.5, 8.0, 14.4, 8.45, "top-k songs", color=C["output_b"])

# Claude → AI Explanation
arrow(ax, 13.5, 7.2, 14.4, 6.7, "explanation text", color=C["agent_b"])

# Ranked Songs → Human Review
arrow(ax, 15.95, 7.8, 15.95, 5.1, "", color=C["test_b"])

# AI Explanation → Human Review
arrow(ax, 15.95, 5.8, 15.95, 5.1, "", color=C["test_b"])

# Test suite → Scoring Engine (unit tests)
arrow(ax, 3.5, 1.8, 4.3, 5.25, "unit tests", color=C["test_b"])
# Test suite → RAG Retriever
arrow(ax, 3.5, 2.1, 4.3, 8.15, "unit tests", color=C["test_b"])

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend_items = [
    (C["rag_b"],    "RAG knowledge retrieval"),
    (C["agent_b"],  "Agentic / Claude API"),
    (C["engine_b"], "Scoring engine"),
    (C["eval_b"],   "Fit evaluator (self-check)"),
    (C["test_b"],   "Testing / human review"),
    (C["data_b"],   "Data source"),
]
lx, ly = 0.5, 0.35
ax.text(lx, ly, "Legend:", fontsize=8, color=C["text_s"], va="bottom")
for i, (color, label) in enumerate(legend_items):
    xi = lx + 0.9 + i * 2.85
    patch = mpatches.Rectangle((xi - 0.35, ly - 0.22), 0.28, 0.2,
                                 fc=color, ec="none", zorder=6)
    ax.add_patch(patch)
    ax.text(xi, ly - 0.12, label, fontsize=7.2, color=C["text_s"],
            va="center", zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
out = "assets/system_architecture.png"
fig.savefig(out, dpi=180, bbox_inches="tight",
            facecolor=C["bg"], edgecolor="none")
plt.close(fig)
print(f"Saved → {out}")
