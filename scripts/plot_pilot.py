"""Pilot summary figure: learning curves (2.4M-step runs) + message-intervention effects."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

R = "experiments/runs"
OUT = "experiments/results/pilot_summary.png"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.1, 1]})

groups = [("c5_ind_long", 0, "selfish, b=0", C[0]), ("c5_ind_long", 1, "selfish, b=1", C[1]),
          ("c5_ind_long", 2, "selfish, b=2", C[2]), ("c5_team_long", 1, "team diagnostic, b=1", C[3])]
for g, b, lab, col in groups:
    for s, ls in [(0, "-"), (1, "--")]:
        p = f"{R}/{g}/b{b}_s{s}/train_log.csv"
        if not os.path.exists(p):
            continue
        lg = pd.read_csv(p)
        y = lg.mean_return.rolling(25, min_periods=1).mean()
        ax[0].plot(lg.env_steps / 1e6, y, color=col, lw=2, ls=ls, label=f"{lab} (seed {s})")
ev = pd.read_csv("experiments/results/c5_ind_long/eval_all.csv")
for name, lab in [("oracle", "full-info oracle"), ("always_gas", "always GAS")]:
    v = ev[(ev.kind == "scripted") & (ev.policy == name)].mean_return.iloc[0]
    ax[0].axhline(v, color=MUTED, lw=1, ls=":")
    ax[0].text(1.6, v + 0.02, f"{lab} ({v:.2f})", color=MUTED, fontsize=8)
ax[0].set(xlabel="environment steps (millions)", ylabel="mean return per car (training episodes, smoothed)",
          title="Learning curves, collision penalty −5, vision 0")
ax[0].grid(axis="y", color=GRID, lw=0.8)
ax[0].set_ylim(-0.6, 0.75)
ax[0].legend(fontsize=7, frameon=False, loc="lower right")

# interventions
rows = []
for g, b, lab in [("c5_ind_long", 2, "selfish b=2"), ("c5_team_long", 1, "team b=1")]:
    d = pd.read_csv(f"experiments/results/{g}/eval_all.csv")
    d = d[(d.kind == "learned") & (d.bits == b)]
    for s in sorted(d.seed.unique()):
        for iv in ["none", "mute", "random", "permute"]:
            r = d[(d.seed == s) & (d.intervention == iv)].iloc[0]
            rows.append(dict(cond=f"{lab}\nseed {s}", iv=iv, coll=100 * r.collision_ep_rate))
df = pd.DataFrame(rows)
conds = list(dict.fromkeys(df.cond))
w = 0.2
for k, iv in enumerate(["none", "mute", "random", "permute"]):
    sub = df[df.iv == iv].set_index("cond").loc[conds]
    xs = [i + (k - 1.5) * w for i in range(len(conds))]
    ax[1].bar(xs, sub.coll, width=w - 0.03, color=C[k], label=iv)
ax[1].set_xticks(range(len(conds)), conds)
ax[1].set(ylabel="episodes with a collision (%)", title="Receiver-side message interventions (1,944 episodes each)")
ax[1].grid(axis="y", color=GRID, lw=0.8)
ax[1].legend(title="intervention", fontsize=8, title_fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("saved", OUT)
