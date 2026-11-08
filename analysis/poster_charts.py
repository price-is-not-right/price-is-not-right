"""
Bar charts for the ICRA poster.
Data taken from Tables I and II of the paper.

Run:  python analysis/poster_charts.py
Outputs:
    chart_singlemove.png, chart_3block.png, chart_4block.png   (success, one per task)
    chart_energy.png                                           (training energy)
    chart_infer.png                                            (inference energy)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 13,
    "font.family": "DejaVu Sans",   # swap for "Arial" to match the poster
    "axes.edgecolor": "#555",
    "axes.linewidth": 0.8,
    "xtick.color": "#333",
    "ytick.color": "#333",
    "text.color": "#222",
    "axes.labelcolor": "#222",
})

# Palette: the two VLAs in blue, the NSM (ours) in an amber accent so it pops
C_E2E = "#3D6FB4"   # E2E-VLA  (blue)
C_PG  = "#9FC0E8"   # PG-VLA   (light blue)
C_NSM = "#E08A2B"   # NSM/ours (amber accent)

MODELS = ["E2E-VLA", "PG-VLA", "NSM (ours)"]
COLORS = [C_E2E, C_PG, C_NSM]


# ---------- Success rate: one separate chart per task ----------
def success_chart(values, title, outfile, subtitle=None):
    """Single chart of the three models' success rate on one task."""
    fig, ax = plt.subplots(figsize=(4.0, 4.4))
    bars = ax.bar(MODELS, values, color=COLORS, width=0.6)
    ax.set_ylabel("Task success (%)")
    ax.set_ylim(0, 108)
    ax.set_title(title, fontsize=15, fontweight="bold", color="#13294B", pad=18)
    if subtitle:
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=10.5, color="#5A6370")
    for r, v in zip(bars, values):
        ax.annotate(f"{v:.0f}", (r.get_x() + r.get_width() / 2, v + 1.5),
                    ha="center", va="bottom", fontsize=12, color="#333")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_success_charts():
    # values are [E2E-VLA, PG-VLA, NSM (ours)]
    success_chart(
        [87.0, 59.6, 99.0],
        "Single-move execution",
        "chart_singlemove.png",
        subtitle="single pick-and-place steps from the 3-block task",
    )
    success_chart(
        [34.0, 0.0, 95.0],
        "3-Block Towers of Hanoi",
        "chart_3block.png",
    )
    success_chart(
        [0.0, 0.0, 78.0],
        "4-Block Towers of Hanoi",
        "chart_4block.png",
    )


# ---------- Total training energy (log scale) ----------
def chart_energy():
    energy = [68.5, 64.9, 0.85]            # MJ, total (GPU + CPU)
    fig, ax = plt.subplots(figsize=(4.0, 4.4))
    bars = ax.bar(MODELS, energy, color=COLORS, width=0.6)
    # ax.set_yscale("log")
    ax.set_ylabel("Energy (MJ)")
    ax.set_ylim(0, 75)
    ax.set_title("Total Training Energy", fontsize=15, fontweight="bold", color="#13294B", pad=18)
    for r, e in zip(bars, energy):
        ax.annotate(f"{e:g} MJ", (r.get_x() + r.get_width() / 2, e+1),
                    ha="center", va="bottom", fontsize=11.5, color="#333")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.7, which="both")
    ax.set_axisbelow(True)
    # ax.annotate("~80x", xy=(2, 0.85), xytext=(1.5, 30),
    #             fontsize=14, color=C_NSM, fontweight="bold")
    fig.tight_layout()
    fig.savefig("chart_energy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- Per-episode inference energy, 3-block ----------
def chart_infer():
    kj = [1.59, 1.41, 0.12]                # kJ per individual move
    fig, ax = plt.subplots(figsize=(4.0, 4.4))
    bars = ax.bar(MODELS, kj, color=COLORS, width=0.6)
    ax.set_ylabel("Energy (kJ)")
    ax.set_ylim(0, 1.75)
    ax.set_title("Energy per Individual Move", fontsize=15, fontweight="bold", color="#13294B", pad=18)
    for r, e in zip(bars, kj):
        ax.annotate(f"{e:g}", (r.get_x() + r.get_width() / 2, e + 0.05),
                    ha="center", va="bottom", fontsize=11.5, color="#333")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("chart_infer.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_success_charts()
    chart_energy()
    chart_infer()
    print("Wrote chart_singlemove.png, chart_3block.png, chart_4block.png, "
          "chart_energy.png, chart_infer.png")
