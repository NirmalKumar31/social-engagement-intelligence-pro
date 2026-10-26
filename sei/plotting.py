"""Shared matplotlib style."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C


def setup():
    plt.rcParams.update({
        "figure.facecolor": C.INK["surface"], "axes.facecolor": C.INK["surface"],
        "savefig.facecolor": C.INK["surface"], "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": C.INK["axis"], "axes.labelcolor": C.INK["secondary"],
        "axes.titlecolor": C.INK["primary"], "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": C.INK["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": C.INK["muted"], "ytick.color": C.INK["muted"], "xtick.labelcolor": C.INK["secondary"],
        "ytick.labelcolor": C.INK["secondary"], "lines.linewidth": 2, "legend.frameon": False,
        "legend.labelcolor": C.INK["secondary"], "figure.dpi": 110, "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def save(fig, name: str):
    fig.savefig(C.FIGURES / f"{name}.png")
    plt.close(fig)
