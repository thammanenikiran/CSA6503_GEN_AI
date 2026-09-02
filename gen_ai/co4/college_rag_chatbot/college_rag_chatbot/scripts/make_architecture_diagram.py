import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

fig, ax = plt.subplots(figsize=(9.5, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 15.5)
ax.axis("off")

COLOR_DATA = "#dbe9f4"
COLOR_PROC = "#1f4e5f"
COLOR_PROC_TEXT = "white"
COLOR_STORE = "#2f9e6f"
COLOR_UI = "#f4a340"
EDGE = "#12313b"

def box(cx, cy, w, h, text, fc, tc="black", fontsize=11, bold=True, edge=EDGE, sub=None):
    b = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle="round,pad=0.06,rounding_size=0.12",
                        linewidth=1.6, edgecolor=edge, facecolor=fc, zorder=2)
    ax.add_patch(b)
    weight = "bold" if bold else "normal"
    if sub:
        ax.text(cx, cy + 0.14, text, ha="center", va="center", fontsize=fontsize,
                 color=tc, weight=weight, zorder=3)
        ax.text(cx, cy - 0.30, sub, ha="center", va="center", fontsize=fontsize-2.3,
                 color=tc, style="italic", zorder=3)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
                 color=tc, weight=weight, zorder=3)

def arrow(x1, y1, x2, y2, text=None, color=EDGE):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.6, color=color, zorder=1, shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if text:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.35, my, text, ha="left", va="center", fontsize=9.2,
                 color="#333333", style="italic")

# Title
ax.text(5, 15.1, "RAG-Based College Chatbot — Architecture", ha="center", va="center",
        fontsize=16, weight="bold", color="#1f4e5f")

# --- Layer 1: Knowledge base ---
box(5, 14.0, 5.4, 0.85, "documents/*.txt", COLOR_DATA, "black", 12,
    sub="6 college documents: admissions, fees, courses, exams,\nhostel/campus life, placements")

arrow(5, 13.55, 5, 12.85, "INGESTION")

# --- Layer 2: Chunking ---
box(5, 12.4, 6.2, 0.95, "Section-Aware Chunker", COLOR_PROC, COLOR_PROC_TEXT, 12,
    sub="chunk_document() — paragraph + heading aware, ~90 words/chunk")

arrow(5, 11.9, 5, 11.2, "INDEXING")

# --- Layer 3: Retrieval index ---
box(5, 10.75, 6.2, 0.95, "TfidfRetriever  (scikit-learn)", COLOR_PROC, COLOR_PROC_TEXT, 12,
    sub="TF-IDF vectors (1–2 grams) + cosine similarity")

# User question enters here
box(1.05, 9.35, 2.0, 0.8, "User\nQuestion", COLOR_UI, "black", 10.5)
arrow(1.75, 9.0, 3.5, 9.85, "query")

arrow(5, 10.25, 5, 9.55, "RETRIEVAL (top-k, score-gated)")

# --- Layer 4: Retrieved chunks ---
box(5, 9.05, 6.2, 0.95, "Top-k Relevant Chunks", COLOR_DATA, "black", 12,
    sub="each tagged with source file + section heading")

ax.text(5.35, 8.2, "AUGMENTATION", ha="left", fontsize=9.2, style="italic", color="#333333")

# --- Layer 5: Generation ---
box(3.35, 7.35, 4.3, 1.15, "ExtractiveSynthesisGenerator", COLOR_PROC, COLOR_PROC_TEXT, 11,
    sub="default, offline — re-ranks & stitches\nsentences from retrieved text")
box(7.7, 7.35, 3.4, 1.15, "LLMGenerator\n(optional)", COLOR_PROC, COLOR_PROC_TEXT, 11,
    sub="hosted LLM call, used if\nOPENAI_API_KEY is set")
ax.text(5.5, 8.02, "GENERATION", ha="center", fontsize=9.2, style="italic", color="#333333")
arrow(5, 8.55, 3.35, 7.95)
arrow(5, 8.55, 7.7, 7.95)

arrow(3.35, 6.75, 5, 6.05)
arrow(7.7, 6.75, 5, 6.05)

# --- Layer 6: Response ---
box(5, 5.55, 6.4, 0.95, "Answer + Cited Sources + Status", COLOR_STORE, "white", 12,
    sub='status: ok · no_match · empty_input · invalid_input · too_long')

arrow(5, 5.05, 5, 4.35, "JSON response")

# --- Layer 7: API / UI ---
box(3.3, 3.85, 4.0, 0.85, "Flask REST API", COLOR_PROC, COLOR_PROC_TEXT, 11.5,
    sub="/api/chat  ·  /api/health")
box(7.4, 3.85, 3.8, 0.85, "Chat UI", COLOR_UI, "black", 11.5,
    sub="HTML / CSS / JS")
arrow(4.6, 3.85, 5.5, 3.85)
arrow(3.3, 3.4, 3.3, 2.7)
box(3.3, 2.25, 4.0, 0.7, "Automated Test Suite", "#e7e1f2", "black", 10.8,
    sub="test_chatbot.py — 19 cases")

# Legend
legend_items = [("Data", COLOR_DATA), ("Processing", COLOR_PROC), ("Response/Store", COLOR_STORE), ("UI / Input", COLOR_UI)]
lx, ly = 0.6, 0.9
ax.text(lx - 0.05, ly + 0.45, "Legend", fontsize=10, weight="bold")
for i, (label, color) in enumerate(legend_items):
    yy = ly - i * 0.32
    ax.add_patch(FancyBboxPatch((lx, yy - 0.09), 0.32, 0.22, boxstyle="round,pad=0.02",
                                 facecolor=color, edgecolor=EDGE, linewidth=1))
    tc = "white" if color == COLOR_PROC else "black"
    ax.text(lx + 0.5, yy + 0.01, label, fontsize=9.5, va="center")

plt.tight_layout()
plt.savefig("/home/claude/college_rag_chatbot/screenshots/03_architecture_diagram.png",
            dpi=200, bbox_inches="tight", facecolor="white")
print("saved")
