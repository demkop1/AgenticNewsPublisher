"""Draws docs/graph-diagram.png with matplotlib (the structure is written out by hand here,
not exported from LangGraph). Run from the repository root: python docs/draw_graph_diagram.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

NODE, NODE_EDGE, TEXT = "#02264b", "#1b4a7a", "#8fc4ff"
DARK, BOX, LINE = "#00101f", "#222222", "#1b4a7a"

fig, ax = plt.subplots(figsize=(7.5, 15))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis("off")


def box(x0, y0, x1, y1, title):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.3",
                                fc=BOX, ec=BOX, zorder=1))
    ax.text(x0 + 0.4, y1 - 0.45, title, color="white", ha="left", va="center", fontsize=11, zorder=2)


def node(x, y, label, w=3.4, h=0.8, dark=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.3",
                                fc=DARK if dark else NODE, ec=NODE_EDGE, lw=1.2,
                                ls=":" if dark else "-", zorder=3))
    ax.text(x, y, label, color=TEXT, ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)


def terminal(x, y, label, r=0.55):
    ax.add_patch(Circle((x, y), r, fc=NODE, ec=NODE_EDGE, lw=1.2, zorder=3))
    ax.text(x, y, label, color=TEXT, ha="center", va="center", fontsize=10, fontweight="bold", zorder=4)


def pill(x, y, label, w=None):
    w = w or 0.2 + 0.125 * len(label)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.25), w, 0.5, boxstyle="round,pad=0,rounding_size=0.25",
                                fc=DARK, ec=NODE_EDGE, lw=1, zorder=5))
    ax.text(x, y, label, color=TEXT, ha="center", va="center", fontsize=9, fontweight="bold", zorder=6)


def arrow(points, dashed=False):
    style = (0, (3, 3)) if dashed else "-"
    xs, ys = zip(*points[:-1])
    ax.plot(xs, ys, color=LINE, lw=1.4, ls=style, zorder=2)
    ax.annotate("", xy=points[-1], xytext=points[-2], zorder=2,
                arrowprops=dict(arrowstyle="->", color=LINE, lw=1.4, ls=style, shrinkA=0, shrinkB=0))


CX = 5
# Search graph
box(0.4, 6.8, 9.6, 18.0, "Search Graph")
terminal(CX, 19.2, "START")
arrow([(CX, 18.65), (CX, 16.9)])

node(CX, 16.5, "Search")
node(CX, 15.1, "Load Published")
node(CX, 13.7, "Fetch Articles")
node(CX, 12.3, "Check Relevance", w=3.0)
node(CX, 10.4, "Create Events")
node(CX, 9.0, "Deduplicate")
node(CX, 7.6, "Store")

arrow([(CX, 16.1), (CX, 15.5)])
arrow([(CX, 14.7), (CX, 14.1)])
arrow([(CX, 13.3), (CX, 12.7)])
arrow([(CX, 11.9), (CX, 10.8)])
pill(CX, 11.35, "Some relevant", w=1.8)
arrow([(CX, 10.0), (CX, 9.4)])
arrow([(CX, 8.6), (CX, 8.0)])

# None relevant -> Critic -> back to Search
node(1.45, 12.3, "Critic", w=1.7, dark=True)
arrow([(CX - 1.5, 12.3), (2.3, 12.3)])
pill(2.85, 11.55, "None relevant", w=1.7)
arrow([(1.45, 12.7), (1.45, 16.5), (CX - 1.7, 16.5)])
pill(1.45, 14.5, "New query", w=1.5)

# Retries used up -> END
terminal(8.65, 12.3, "END", r=0.5)
arrow([(CX + 1.5, 12.3), (8.15, 12.3)], dashed=True)
pill(7.3, 11.55, "Retries used up", w=1.9)

# Hand-off to the publisher graph
arrow([(CX, 6.8), (CX, 6.45)])
node(CX, 6.0, "Events found?", w=2.6, h=0.7, dark=True)
box(1.6, 1.7, 8.4, 5.2, "Publisher Graph")
arrow([(CX, 5.65), (CX, 4.3)], )
node(CX, 3.9, "Rerank Events")
node(CX, 2.5, "Generate Article")
arrow([(CX, 3.5), (CX, 2.9)])

terminal(CX, 0.65, "END")
arrow([(CX, 2.1), (CX, 1.2)])
arrow([(CX + 1.3, 6.0), (9.0, 6.0), (9.0, 0.65), (CX + 0.55, 0.65)], dashed=True)
pill(9.0, 3.3, "No events", w=1.5)

fig.savefig("docs/graph-diagram.png", dpi=140, bbox_inches="tight", pad_inches=0.15, facecolor="white")
