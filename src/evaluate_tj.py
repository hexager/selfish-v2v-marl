"""Evaluate trained populations or scripted drivers on the traffic junction.

python -m src.evaluate_tj --runs experiments/runs/tj/medium_b0_s0 experiments/runs/tj/medium_b1_s0 \
    --episodes 500 --out experiments/results/tj_medium

Interventions edit only what a receiver hears (env.inbox_transform); senders and physics are unchanged.
  mute    every heard symbol -> 0
  random  every heard symbol -> uniform random symbol (drawn per receiver and sender)
  permute every heard symbol s -> s+1 mod 2^b
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.envs.traffic_junction import TrafficJunctionEnv
from src.evaluate import mutual_information
from src.viz.record import PopulationDriver, ScriptedDriver

def make_transform(name, n_sym):
    if name == "none" or n_sym == 0:
        return None
    if name == "mute":
        return lambda i, j, s, rng: 0
    if name == "random":
        return lambda i, j, s, rng: int(rng.integers(n_sym))
    if name == "permute":
        return lambda i, j, s, rng: (s + 1) % n_sym
    raise ValueError(name)


def gini(x):
    """Gini coefficient of non-negative values (here: each car's cost = -return, since TJ rewards are <= 0)."""
    x = np.sort(np.clip(np.asarray(x, float), 0, None))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def evaluate(env, driver, episodes, seed0, add_rate, intervention="none", log_messages=False):
    env.inbox_transform = make_transform(intervention, env.n_sym)
    env.rng_eval = np.random.default_rng(seed0 + 7)
    rows, msgs = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep, options={"add_rate": add_rate})
        driver.reset(env)
        car_ret, rets, exit_tau = {}, [], []
        while env.agents:
            acts, _ = driver.act(env, obs)
            if log_messages and env.n_sym:
                for i, a in enumerate(env.possible_agents):
                    if a in acts:
                        act = np.asarray(acts[a]).reshape(-1)
                        msgs.append(dict(route=int(env.path_id[i]), move=int(act[0]), msg=int(act[1]),
                                         prog=int(env.prog[i]), tau=int(env.tau[i])))
            obs, rew, te, tr, info = env.step(acts)
            exit_tau += env.exit_tau
            for i, a in enumerate(env.possible_agents):
                inf = info[a]
                if inf["acted"]:                                   # the car that acted this step
                    car_ret[i] = car_ret.get(i, 0.0) + inf["r_acted"]
                    if inf["car_done"]:
                        rets.append(car_ret.pop(i))
                if inf["spawned"]:                                 # a new car in this slot
                    car_ret[i] = inf["r_spawn"]
        rets += list(car_ret.values())                             # cars still on the grid at the end
        st = env.stats
        rows.append(dict(episode=ep, success=1 - st["failed"], spawned=st["spawned"], completed=st["completed"],
                         cars_collided=st["cars_collided"], mean_car_return=float(np.mean(rets)) if rets else 0.0,
                         worst_car_return=float(np.min(rets)) if rets else 0.0, cost_gini=gini(-np.asarray(rets)) if rets else 0.0,
                         mean_exit_tau=float(np.mean(exit_tau)) if exit_tau else np.nan))
    env.inbox_transform = None
    d = pd.DataFrame(rows)
    summary = dict(episodes=len(d), success_rate=d.success.mean(),
                   car_collision_rate=d.cars_collided.sum() / max(d.spawned.sum(), 1),
                   completion_rate=d.completed.sum() / max(d.spawned.sum(), 1),
                   mean_car_return=d.mean_car_return.mean(), worst_car_return=d.worst_car_return.mean(),
                   cost_gini=d.cost_gini.mean(), mean_exit_tau=d.mean_exit_tau.mean(),
                   cars_per_episode=d.spawned.mean())
    if msgs:
        m = pd.DataFrame(msgs)
        p = m.msg.value_counts(normalize=True).values
        summary.update(msg_entropy=float(-(p * np.log2(p)).sum()), mi_msg_route=mutual_information(m.msg, m.route),
                       mi_msg_move=mutual_information(m.msg, m.move))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--config", default="configs/tj_medium.yaml", help="for scripted baselines")
    ap.add_argument("--scripted", nargs="*", default=["always_gas", "oracle"])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=100000, help="evaluation seeds start here (training never uses them)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from src.train_tj import load_policies
    import yaml
    rows = []
    base_cfg = yaml.safe_load(open(a.config))
    add_rate = base_cfg["curriculum"]["add_rate_end"]
    for name in a.scripted:
        env = TrafficJunctionEnv(bits=0, **base_cfg["env"])
        s = evaluate(env, ScriptedDriver(name), a.episodes, a.seed, add_rate)
        rows.append(dict(policy=name, bits=0, seed=-1, intervention="none", **s))
        print(name, {k: round(v, 3) for k, v in s.items()})
    for run in a.runs:
        cfg = json.load(open(os.path.join(run, "manifest.json")))
        pols, ck = load_policies(run)
        env_kw = dict(cfg["config"]["env"])
        env_kw["alpha"] = 0.0          # report each car's own outcome, whatever reward the policy was trained on
        env = TrafficJunctionEnv(bits=ck["bits"], **env_kw)
        ivs = ["none"] if ck["bits"] == 0 else ["none", "mute", "random", "permute"]
        for iv in ivs:
            drv = PopulationDriver(pols, ck["pop_mode"], seed=a.seed)
            s = evaluate(env, drv, a.episodes, a.seed, cfg["config"]["curriculum"]["add_rate_end"], iv,
                         log_messages=(iv == "none"))
            rows.append(dict(policy=os.path.basename(run), bits=ck["bits"], seed=cfg["seed"], intervention=iv, **s))
            print(os.path.basename(run), iv, {k: round(v, 3) for k, v in s.items()})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "eval.csv"), index=False)


if __name__ == "__main__":
    main()
