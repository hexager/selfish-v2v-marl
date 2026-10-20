"""Learning curves for traffic-junction runs.
python scripts/plot_tj.py --runs experiments/runs/tj/medium_b0_s0 ... --labels "v0 b=0" ... --out experiments/results/tj_curves.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

C = ["#2a78d6", "#eb6834", "#1baf7a", "#d55181", "#d99500", "#6f55c9"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--labels", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--smooth", type=int, default=25)
a = ap.parse_args()
labels = a.labels or [os.path.basename(r) for r in a.runs]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for k, (run, lab) in enumerate(zip(a.runs, labels)):
    lg = pd.read_csv(os.path.join(run, "train_log.csv"))
    x = lg.env_steps / 1e6
    for j, col in enumerate(["success", "car_collision_rate", "car_return"]):
        ax[j].plot(x, lg[col].rolling(a.smooth, min_periods=1).mean(), color=C[k % len(C)], lw=1.8, label=lab)
    if k == 0:
        ax2 = ax[0].twinx()
        ax2.plot(x, lg.add_rate, color=MUTED, lw=1, ls=":")
        ax2.set_ylabel("arrival rate (dotted)", color=MUTED)
        ax2.spines["top"].set_visible(False)
titles = ["Episodes with no collision", "Cars that collided", "Return per car"]
ylabels = ["success rate (training episodes)", "fraction of cars", "mean return"]
for j in range(3):
    ax[j].set(title=titles[j], xlabel="environment steps (millions)", ylabel=ylabels[j])
    ax[j].grid(axis="y", color=GRID, lw=0.8)
ax[2].legend(fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(a.out, dpi=140)
print("saved", a.out)
