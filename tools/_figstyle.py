"""Shared publication figure style for P1 + P2 (top-tier venue standard).

Design contract (per request): keep the panels clean — axes, data, a short
legend, and only the value labels that carry information. All descriptive
sentences live in the LaTeX caption ("figure footnote"), NOT in the panel.

- Okabe-Ito colorblind-safe palette (safe in grayscale too).
- Serif type to match the paper body; STIX math to pair with Times.
- Despined (no top/right), thin axes, no heavy grid.
- Vector PDF with editable text (pdf.fonttype 42).

Import and call apply_style() at the top of every figure script.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# Prefer a real Times-metric serif so panels match the \usepackage{times} body.
# Times New Roman / Nimbus Roman are rarely installed on HPC; STIXGeneral is the
# free Times-compatible face that also pairs with the STIX math set (single
# consistent family for text + math). Fall back to DejaVu Serif only if absent.
_AVAIL = {f.name for f in _fm.fontManager.ttflist}
_SERIF = [f for f in ("Times New Roman", "Nimbus Roman", "TeX Gyre Termes",
                      "STIXGeneral", "DejaVu Serif") if f in _AVAIL] or ["DejaVu Serif"]

# --- Okabe-Ito colorblind-safe palette --------------------------------------
BLUE   = "#0072B2"   # primary
ORANGE = "#E69F00"
GREEN  = "#009E73"
VERM   = "#D55E00"   # emphasis / "bad" direction
PURPLE = "#CC79A7"
SKY    = "#56B4E9"
YELLOW = "#F0E442"
GREY   = "#8C8C8C"
INK    = "#222222"


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 200, "savefig.dpi": 600,
        "font.family": "serif",
        "font.serif": _SERIF,
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.titlesize": 10.5, "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8, "axes.edgecolor": "#333333",
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.grid": False,
        "legend.frameon": False, "legend.handlelength": 1.5,
        "legend.columnspacing": 1.1, "legend.labelspacing": 0.35,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def despine(ax):
    ax.spines[["top", "right"]].set_visible(False)


def light_grid(ax, axis="y"):
    ax.grid(axis=axis, ls=":", lw=0.6, alpha=0.35, color="#B0B0B0")
    ax.set_axisbelow(True)
