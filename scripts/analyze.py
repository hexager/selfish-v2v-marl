"""Evaluate all runs in a sweep dir: baselines, interventions, message statistics, cross-play, plots.
python scripts/analyze.py --runs experiments/runs/pilot --out experiments/results/pilot
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, ".")
from src.evaluate import evaluate, mutual_information  # noqa: E402
from src.train import load_policies  # noqa: E402

KEYS = ["collision_ep_rate", "arrival_rate", "timeout_rate", "mean_ticks_arrived", "mean_return",
        "worst_car_return", "welfare"]


def env_kwargs(run):
    cfg = json.load(open(os.path.join(run, "manifest.json")))["config"]
    return dict(**cfg["env"], **cfg["reward"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = sorted(glob.glob(os.path.join(a.runs, "b*_s*")))
    kw0 = env_kwargs(runs[0])

    rows = []
    for name in ["random", "always_gas", "local_right_of_way", "oracle"]:
        m, _, _ = evaluate({**kw0, "bits": 0}, scripted=name, reps=a.reps)
        rows.append(dict(kind="scripted", policy=name, bits=0, seed=-1, intervention="none", **m))

    msg_rows = []
    for run in runs:
        pols, bits = load_policies(run)
        seed = int(run.split("_s")[-1])
        kw = {**env_kwargs(run), "bits": bits}
        for iv in (["none"] if bits == 0 else ["none", "mute", "random", "permute"]):
            m, _, msgs = evaluate(kw, policies=pols, reps=a.reps, intervention=iv,
                                  record_messages=(iv == "none"))
            rows.append(dict(kind="learned", policy="ippo", bits=bits, seed=seed, intervention=iv, **m))
            if iv == "none" and msgs:
                d = pd.DataFrame(msgs)
                for car in sorted(d.car.unique()):
                    dc = d[d.car == car]
                    p = dc.msg.value_counts(normalize=True).values
                    msg_rows.append(dict(bits=bits, seed=seed, car=car,
                                         msg_entropy=float(-(p * np.log2(p)).sum()),
                                         n_symbols_used=int((p > 0.01).sum()),
                                         mi_msg_route=mutual_information(dc.msg, dc.route),
                                         mi_msg_move=mutual_information(dc.msg, dc.move),
                                         mi_msg_approach=mutual_information(dc.msg, dc.approach)))
        print("evaluated", run, flush=True)

    # cross-play: car_0 from seed i with car_1 from seed j (i != j), same bits
    by_bits = {}
    for run in runs:
        pols, bits = load_policies(run)
        by_bits.setdefault(bits, []).append((run, pols))
    for bits, lst in by_bits.items():
        for i, (ri, pi) in enumerate(lst):
            for j, (rj, pj) in enumerate(lst):
                if i == j:
                    continue
                m, _, _ = evaluate({**env_kwargs(ri), "bits": bits}, policies=[pi[0], pj[1]], reps=1)
                rows.append(dict(kind="crossplay", policy="ippo", bits=bits,
                                 seed=int(f"{ri.split('_s')[-1]}{rj.split('_s')[-1]}"),
                                 intervention="none", **m))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "eval_all.csv"), index=False)
    mdf = pd.DataFrame(msg_rows)
    mdf.to_csv(os.path.join(a.out, "message_stats.csv"), index=False)

    learned = df[df.kind == "learned"]
    agg = learned.groupby(["bits", "intervention"])[KEYS].agg(["mean", "std", "count"])
    agg.to_csv(os.path.join(a.out, "summary_by_bits.csv"))
    cp = df[df.kind == "crossplay"].groupby("bits")[KEYS].agg(["mean", "std", "count"])
    cp.to_csv(os.path.join(a.out, "crossplay_by_bits.csv"))
    if len(mdf):
        mdf.groupby("bits")[["msg_entropy", "n_symbols_used", "mi_msg_route", "mi_msg_move",
                             "mi_msg_approach"]].agg(["mean", "std"]).to_csv(
            os.path.join(a.out, "message_stats_by_bits.csv"))

    # learning curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    colors = {0: "#4c4c4c", 1: "#1f77b4", 2: "#2ca02c", 4: "#d62728"}
    for run in runs:
        bits = int(os.path.basename(run).split("_")[0][1:])
        lg = pd.read_csv(os.path.join(run, "train_log.csv"))
        sm = lg[["mean_return", "collision_ep_rate"]].rolling(10, min_periods=1).mean()
        ax[0].plot(lg.env_steps, sm.mean_return, color=colors.get(bits), alpha=.7,
                   label=f"b={bits}" if run.endswith("_s0") else None)
        ax[1].plot(lg.env_steps, sm.collision_ep_rate, color=colors.get(bits), alpha=.7)
    for name, ls in [("always_gas", ":"), ("local_right_of_way", "--"), ("oracle", "-.")]:
        r = df[(df.kind == "scripted") & (df.policy == name)].iloc[0]
        ax[0].axhline(r.mean_return, color="k", ls=ls, lw=1, label=name)
        ax[1].axhline(r.collision_ep_rate, color="k", ls=ls, lw=1)
    ax[0].set(xlabel="env steps", ylabel="mean return per car (train episodes)", title="Return")
    ax[1].set(xlabel="env steps", ylabel="fraction of episodes with a collision", title="Collisions")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "learning_curves.png"), dpi=130)
    print(agg[[("collision_ep_rate", "mean"), ("mean_return", "mean"), ("arrival_rate", "mean")]])


if __name__ == "__main__":
    main()
