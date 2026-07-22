"""
Shared figure style for the results-section figures.

Every plotting script imports from here, so the whole paper's look — fonts,
sizes, colours — changes in ONE place. Plotting is deliberately separated from
the heavy computation: the cluster writes a small *_data.npz (a few KB), and the
local fig_*_plot.py scripts render it, so titles / axes / colours can be tweaked
instantly with no cluster and no big files.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- class + accent colours (validated colourblind-safe palette) ------------- #
C_E, C_NONE, C_I = "#2a78d6", "#898781", "#e34948"      # excitatory / none / inhib
ACCENT, ACCENT2 = "#2a78d6", "#4a3aa7"                  # curves, operating point
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
CLASS_COLOR = {"E": C_E, "none": C_NONE, "I": C_I}

# single-hue blue ramp for confusion heatmaps
BLUE_SEQ = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fe", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95"])


def apply_style():
    """Global rcParams — call once at the top of every plot script."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
        "axes.labelcolor": MUTED, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text in the vector file
    })


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def save(fig, stem: str | Path, dpi: int = 200):
    """Save both a vector PDF (for the journal) and a PNG preview."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=dpi)
    print(f"wrote {stem.with_suffix('.pdf')} and .png")
