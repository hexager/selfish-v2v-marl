"""Independent PPO on the traffic junction, with a population of networks.

Every car that arrives is driven by a network drawn at random from a population of K networks
(`population.mode: random`). Each network is trained only on the experience and the rewards of
the cars it drove. `population.mode: fixed` gives slot i always network i (the pilot setting,
kept as the identity-convention ablation).

python -m src.train_tj --config configs/tj_medium.yaml --bits 1 --seed 0 --out experiments/runs/tj
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
import yaml

from src.envs.traffic_junction import TrafficJunctionEnv
from src.rl.ppo import CarPolicy, PPOConfig, gae, ppo_update


def make_env(cfg, bits):
    return TrafficJunctionEnv(bits=bits, **cfg["env"])


def add_rate_at(cfg, frac):
    """Linear curriculum on the arrival rate (IC3Net: constant, then linear, then constant)."""
    c = cfg["curriculum"]
    a0, a1, s, e = c["add_rate_start"], c["add_rate_end"], c["start_frac"], c["end_frac"]
    if frac <= s:
        return a0
    if frac >= e:
        return a1
    return a0 + (a1 - a0) * (frac - s) / (e - s)


def train(cfg, bits, seed, out_dir, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    tc = cfg["train"]
    E, T = tc["num_envs"], tc["rollout_len"]
    envs = [make_env(cfg, bits) for _ in range(E)]
    env0 = envs[0]
    N, D = env0.n_max, env0._obs_dim
    agents = env0.possible_agents
    pop = cfg.get("population", {"mode": "random", "K": 4})
    K = N if pop["mode"] == "fixed" else int(pop["K"])
    rng = np.random.default_rng(seed)
    pols = [CarPolicy(D, env0.n_sym, tc["hidden"]) for _ in range(K)]
    pcfg = PPOConfig(**tc.get("ppo", {}))
    opts = [torch.optim.Adam(p.parameters(), lr=pcfg.lr, eps=1e-5) for p in pols]

    def draw_net(slot):
        return slot if pop["mode"] == "fixed" else int(rng.integers(K))

    n_updates = max(1, tc["total_steps"] // (E * T))
    cur = []
    assign = np.full((E, N), -1)
    car_ret = np.zeros((E, N))
    for e in envs:
        o, _ = e.reset(seed=int(rng.integers(2**31)), options={"add_rate": add_rate_at(cfg, 0.0)})
        cur.append(o)
    ep_records, car_records, log = [], [], []
    t0, steps = time.time(), 0

    for upd in range(n_updates):
        frac = upd / n_updates
        if tc.get("anneal_lr", True):
            for opt in opts:
                opt.param_groups[0]["lr"] = pcfg.lr * (1 - frac)
        shp = (T, E, N)
        B = dict(obs=np.zeros(shp + (D,), np.float32), move=np.zeros(shp, np.int64), msg=np.zeros(shp, np.int64),
                 logp=np.zeros(shp, np.float32), val=np.zeros(shp, np.float32), rew=np.zeros(shp, np.float32),
                 done=np.zeros(shp, np.float32), valid=np.zeros(shp, bool), net=np.full(shp, -1))
        for t in range(T):
            ob = np.zeros((E, N, D), np.float32)
            act = np.zeros((E, N), bool)
            for ei, env in enumerate(envs):
                act[ei] = env.active
                for i, a in enumerate(agents):
                    ob[ei, i] = cur[ei][a]
            net_now = np.where(act, assign, -1)
            mv = np.zeros((E, N), np.int64); mg = np.zeros((E, N), np.int64)
            with torch.no_grad():
                for k in range(K):
                    idx = np.argwhere(net_now == k)
                    if len(idx) == 0:
                        continue
                    x = torch.as_tensor(ob[idx[:, 0], idx[:, 1]])
                    m, g, lp, _, _ = pols[k].act(x)
                    v = pols[k].value(x)
                    mv[idx[:, 0], idx[:, 1]] = m.numpy(); mg[idx[:, 0], idx[:, 1]] = g.numpy()
                    B["logp"][t, idx[:, 0], idx[:, 1]] = lp.numpy()
                    B["val"][t, idx[:, 0], idx[:, 1]] = v.numpy()
            B["obs"][t], B["move"][t], B["msg"][t], B["valid"][t], B["net"][t] = ob, mv, mg, act, net_now
            for ei, env in enumerate(envs):
                acts = {a: (int(mv[ei, i]) if bits == 0 else np.array([mv[ei, i], mg[ei, i]]))
                        for i, a in enumerate(agents) if act[ei, i]}
                o, r, te, tr, info = env.step(acts)
                trunc = not env.agents
                for i, a in enumerate(agents):
                    inf = info[a]
                    if inf["acted"]:
                        B["rew"][t, ei, i] = inf["r_acted"]
                        car_ret[ei, i] += inf["r_acted"]
                        ended = inf["car_done"] or (trunc and inf["active"] and not inf["spawned"])
                        B["done"][t, ei, i] = float(ended or trunc)
                        if ended or (trunc and not inf["spawned"] and inf["active"]):
                            car_records.append(dict(update=upd, ret=car_ret[ei, i], arrived=inf["arrived"],
                                                    net=int(assign[ei, i])))
                            car_ret[ei, i] = 0.0
                    if inf["spawned"]:
                        assign[ei, i] = draw_net(i)
                        car_ret[ei, i] = inf["r_spawn"]          # logged, not trained on (the car had not acted)
                        if trunc:
                            car_ret[ei, i] = 0.0
                cur[ei] = o
                if trunc:
                    st = env.stats
                    ep_records.append(dict(update=upd, success=1 - st["failed"], spawned=st["spawned"],
                                           completed=st["completed"], cars_collided=st["cars_collided"],
                                           add_rate=env.add_rate))
                    assign[ei] = -1
                    car_ret[ei] = 0.0
                    cur[ei], _ = env.reset(seed=int(rng.integers(2**31)),
                                           options={"add_rate": add_rate_at(cfg, frac)})
            steps += E

        # bootstrap values for cars still on the grid
        last_val = np.zeros((E, N), np.float32); last_alive = np.zeros((E, N), np.float32)
        with torch.no_grad():
            for ei, env in enumerate(envs):
                for i, a in enumerate(agents):
                    if env.active[i] and assign[ei, i] >= 0:
                        last_val[ei, i] = float(pols[assign[ei, i]].value(torch.as_tensor(cur[ei][a])[None]))
                        last_alive[ei, i] = 1.0
        adv = np.zeros((T, E, N), np.float32); ret = np.zeros((T, E, N), np.float32)
        for i in range(N):
            adv[:, :, i], ret[:, :, i] = gae(B["rew"][:, :, i], B["val"][:, :, i], B["done"][:, :, i],
                                             B["valid"][:, :, i], last_val[:, i], last_alive[:, i],
                                             pcfg.gamma, pcfg.gae_lambda)
        row = dict(update=upd, env_steps=steps, wall=time.time() - t0)
        flat = {k: v.reshape((-1,) + v.shape[3:]) for k, v in B.items()}
        fadv, fret = adv.reshape(-1), ret.reshape(-1)
        stats = []
        for k in range(K):                     # independent update: network k sees only its own cars
            m = flat["valid"] & (flat["net"] == k)
            if m.sum() < 8:
                continue
            batch = dict(obs=torch.as_tensor(flat["obs"][m]), move=torch.as_tensor(flat["move"][m]),
                         msg=torch.as_tensor(flat["msg"][m]), logp=torch.as_tensor(flat["logp"][m]),
                         adv=torch.as_tensor(fadv[m]), ret=torch.as_tensor(fret[m]))
            stats.append(ppo_update(pols[k], opts[k], batch, pcfg))
        for key in ("ent_move", "ent_msg", "msg_grad_norm", "v_loss", "clipfrac"):
            row[key] = float(np.mean([s[key] for s in stats])) if stats else np.nan
        eps = [r for r in ep_records if r["update"] == upd]
        cars = [r for r in car_records if r["update"] == upd]
        if eps:
            row["episodes"] = len(eps)
            row["success"] = float(np.mean([r["success"] for r in eps]))
            sp = sum(r["spawned"] for r in eps)
            row["car_collision_rate"] = sum(r["cars_collided"] for r in eps) / max(sp, 1)
            row["completion_rate"] = sum(r["completed"] for r in eps) / max(sp, 1)
            row["add_rate"] = float(np.mean([r["add_rate"] for r in eps]))
        if cars:
            row["car_return"] = float(np.mean([c["ret"] for c in cars]))
        log.append(row)
        if verbose and (upd % max(1, n_updates // 20) == 0 or upd == n_updates - 1):
            print(f"[{env0.difficulty} b={bits} s={seed}] upd {upd+1}/{n_updates} steps {steps} "
                  f"succ {row.get('success', np.nan):.3f} carcoll {row.get('car_collision_rate', np.nan):.3f} "
                  f"ret {row.get('car_return', np.nan):.2f} add {row.get('add_rate', np.nan):.3f} "
                  f"Hmsg {row['ent_msg']:.2f} {time.time()-t0:.0f}s", flush=True)

    run = os.path.join(out_dir, f"{env0.difficulty}_b{bits}_s{seed}")
    os.makedirs(run, exist_ok=True)
    pd.DataFrame(log).to_csv(os.path.join(run, "train_log.csv"), index=False)
    torch.save({"bits": bits, "obs_dim": D, "hidden": tc["hidden"], "K": K, "pop_mode": pop["mode"],
                "state_dicts": [p.state_dict() for p in pols]}, os.path.join(run, "policies.pt"))
    with open(os.path.join(run, "manifest.json"), "w") as f:
        json.dump(dict(config=cfg, bits=bits, seed=seed, env_steps=steps, updates=n_updates,
                       wall_seconds=time.time() - t0, python=platform.python_version(),
                       torch=torch.__version__, numpy=np.__version__, machine=platform.machine(),
                       processor=platform.processor(), cpu_threads=1), f, indent=1)
    return pols, run


def load_policies(run_dir):
    ck = torch.load(os.path.join(run_dir, "policies.pt"))
    n_sym = 2 ** ck["bits"] if ck["bits"] > 0 else 0
    pols = []
    for sd in ck["state_dicts"]:
        p = CarPolicy(ck["obs_dim"], n_sym, ck["hidden"])
        p.load_state_dict(sd)
        p.eval()
        pols.append(p)
    return pols, ck


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/runs/tj")
    ap.add_argument("--total_steps", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.total_steps:
        cfg["train"]["total_steps"] = a.total_steps
    train(cfg, a.bits, a.seed, a.out)
