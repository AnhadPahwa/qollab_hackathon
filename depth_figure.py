"""
Average the depth-study runs in ionq_runs.json and draw the figure.

    python depth_figure.py            (writes figures/depth_forte1_average.png)

Each run measures signal survival at 1 to 6 encoding layers in one circuit.
The figure shows the mean of the runs and the lowest and highest value at
each depth. It also fits one number: the signal lost per two-qubit gate.

Gate counting: a block has 6 CZ gates per layer as written, but the two CZ
layers in the middle (the end of the candidate's encoding and the start of
the prototype's inverse) have only diagonal gates between them and cancel.
So the gates that can actually contribute noise number 6 x layers - 6.
"""

import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = [r for r in json.load(open("ionq_runs.json"))["experiment_runs"]
        if r["experiment"] == "depth" and "ionq" in r["backend"]]
layers = runs[0]["layers"]
S = np.array([r["survival"] for r in runs])
mean, lo, hi = S.mean(0), S.min(0), S.max(0)
effective = np.array([6 * d - 6 for d in layers])

# survival = (1 - p) ^ effective_gates, fitted through the origin on a log scale
slope = float((effective * np.log(mean)).sum() / (effective ** 2).sum())
loss = 1 - math.exp(slope)
fit = np.exp(slope * effective)

print(f"{len(runs)} runs on {runs[0]['backend']}, {runs[0]['shots']} shots each")
print("layers  gates that count  mean  lowest  highest  fitted")
for d, g, m, a, b, f in zip(layers, effective, mean, lo, hi, fit):
    print(f"{d:6d} {g:17d} {m:5.0%} {a:7.0%} {b:8.0%} {f:7.0%}")
print(f"fitted loss per two-qubit gate: {loss:.1%}")

INK, MUTED, SURFACE, RULE, BLUE = "#0b0b0b", "#52514e", "#fcfcfb", "#c9c8c2", "#2a78d6"
fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(color="#e6e5e0", linewidth=0.8, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(RULE)
ax.tick_params(colors=MUTED, labelsize=8, length=0)
ax.plot(layers, fit, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2,
        label=f"Fit: {loss:.1%} lost per two-qubit gate")
ax.errorbar(layers, mean, yerr=[mean - lo, hi - mean], color=BLUE, linewidth=2,
            marker="o", markersize=7, capsize=4, elinewidth=1.4, zorder=3,
            label=f"Mean of {len(runs)} runs (bars: lowest to highest)")
for d, m, h in zip(layers, mean, hi):
    ax.annotate(f"{m:.0%}", (d, h), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8, color=MUTED)
ax.set_ylim(0, 1.12)
ax.set_xticks(layers)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Encoding layers", color=MUTED, fontsize=9)
ax.set_ylabel("Signal survived", color=MUTED, fontsize=9)
ax.set_title("Signal surviving the noise, by encoding depth (IonQ Forte 1 simulator)",
             color=INK, fontsize=11, loc="left", pad=12)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower left")
fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/depth_forte1_average.png", dpi=150)
print("wrote figures/depth_forte1_average.png")
