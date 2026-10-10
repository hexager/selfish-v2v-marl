"""Train independent PPO cars on the intersection.

python -m src.train --config configs/pilot_2car.yaml --bits 1 --seed 0 --out experiments/runs
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

from src.envs.intersection import IntersectionEnv
from src.rl.ppo import CarPolicy, PPOConfig, gae, ppo_update


def make_env(cfg, bits):
    return IntersectionEnv(bits=bits, **cfg["env"], **cfg["reward"])


def train(cfg, bits, seed, out_dir, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    tc = cfg["train"]
    E, T = tc["num_envs"], tc["rollout_len"]
    envs = [make_env(cfg, bits) for _ in range(E)]
    rng = np.random.default_rng(seed)
    n, agents = envs[0].n, envs[0].possible_agents
    obs_dim = envs[0]._obs_dim
    pols = [CarPolicy(obs_dim, envs[0].n_sym, tc["hidden"]) for _ in range(n)]
    pcfg = PPOConfig(**tc.get("ppo", {}))
    opts = [torch.optim.Adam(p.parameters(), lr=pcfg.lr, eps=1e-5) for p in pols]

    cur = []
    for e in envs:
        o, _ = e.reset(seed=int(rng.integers(2**31)))
        cur.append(o)
    ep_ret = np.zeros((E, n))
    finished = []          # per-episode records
    log = []
    n_updates = tc["total_steps"] // (E * T)
    t0 = time.time()
    steps = 0
    for upd in range(n_updates):
        if tc.get("anneal_lr", True):
            for opt in opts:
                opt.param_groups[0]["lr"] = pcfg.lr * (1 - upd / n_updates)
        B = {k: np.zeros((T, E, n) + s, d) for k, s, d in
             [("obs", (obs_dim,), np.float32), ("move", (), np.int64), ("msg", (), np.int64),
              ("logp", (), np.float32), ("val", (), np.float32), ("rew", (), np.float32),
              ("done", (), np.float32), ("valid", (), bool)]}
        for t in range(T):
            ob = np.zeros((E, n, obs_dim), np.float32)
            alive = np.zeros((E, n), bool)
            for ei in range(E):
                for i, a in enumerate(agents):
                    if a in envs[ei].agents:
                        ob[ei, i] = cur[ei][a]
                        alive[ei, i] = True
            mv = np.zeros((E, n), np.int64); mg = np.zeros((E, n), np.int64)
            with torch.no_grad():
                for i in range(n):
                    x = torch.as_tensor(ob[:, i])
                    m, g, lp, _, _ = pols[i].act(x)
                    v = pols[i].value(x)
                    mv[:, i], mg[:, i] = m.numpy(), g.numpy()
                    B["logp"][t, :, i], B["val"][t, :, i] = lp.numpy(), v.numpy()
            B["obs"][t], B["move"][t], B["msg"][t], B["valid"][t] = ob, mv, mg, alive
            for ei, env in enumerate(envs):
                acts = {}
                for i, a in enumerate(agents):
                    if alive[ei, i]:
                        acts[a] = int(mv[ei, i]) if bits == 0 else np.array([mv[ei, i], mg[ei, i]])
                o, r, te, tr, info = env.step(acts)
                for i, a in enumerate(agents):
                    if alive[ei, i]:
                        B["rew"][t, ei, i] = r[a]
                        B["done"][t, ei, i] = float(te[a] or tr[a])
                        ep_ret[ei, i] += r[a]
                cur[ei] = o
                if not env.agents:
                    finished.append(dict(update=upd,
                                         ret=ep_ret[ei].tolist(),
                                         crash=env.crashed.tolist(), arrive=env.arrived.tolist(),
                                         ticks=env.ticks.tolist()))
                    ep_ret[ei] = 0
                    cur[ei], _ = env.reset(seed=int(rng.integers(2**31)))
            steps += E
        # bootstrap values for cars still on the road
        last_ob = np.zeros((E, n, obs_dim), np.float32); last_alive = np.zeros((E, n), np.float32)
        for ei in range(E):
            for i, a in enumerate(agents):
                if a in envs[ei].agents:
                    last_ob[ei, i] = cur[ei][a]; last_alive[ei, i] = 1
        row = dict(update=upd, env_steps=steps, wall=time.time() - t0)
        for i in range(n):
            with torch.no_grad():
                lv = pols[i].value(torch.as_tensor(last_ob[:, i])).numpy()
            adv, ret = gae(B["rew"][:, :, i], B["val"][:, :, i], B["done"][:, :, i], B["valid"][:, :, i],
                           lv, last_alive[:, i], pcfg.gamma, pcfg.gae_lambda)
            m = B["valid"][:, :, i].reshape(-1)
            batch = {"obs": torch.as_tensor(B["obs"][:, :, i].reshape(-1, obs_dim)[m]),
                     "move": torch.as_tensor(B["move"][:, :, i].reshape(-1)[m]),
                     "msg": torch.as_tensor(B["msg"][:, :, i].reshape(-1)[m]),
                     "logp": torch.as_tensor(B["logp"][:, :, i].reshape(-1)[m]),
                     "adv": torch.as_tensor(adv.reshape(-1)[m]),
                     "ret": torch.as_tensor(ret.reshape(-1)[m])}
            st = ppo_update(pols[i], opts[i], batch, pcfg)    # car i's own data and own return only
            row.update({f"car{i}_{k}": v for k, v in st.items()})
        recent = [f for f in finished if f["update"] == upd]
        if recent:
            row["episodes"] = len(recent)
            row["mean_return"] = float(np.mean([np.mean(f["ret"]) for f in recent]))
            row["collision_ep_rate"] = float(np.mean([any(f["crash"]) for f in recent]))
            row["arrival_rate"] = float(np.mean([np.mean(f["arrive"]) for f in recent]))
            for i in range(n):
                row[f"car{i}_return"] = float(np.mean([f["ret"][i] for f in recent]))
        log.append(row)
        if verbose and (upd % max(1, n_updates // 10) == 0 or upd == n_updates - 1):
            print(f"[b={bits} s={seed}] upd {upd+1}/{n_updates} steps {steps} ret {row.get('mean_return', np.nan):.3f} "
                  f"coll {row.get('collision_ep_rate', np.nan):.3f} arr {row.get('arrival_rate', np.nan):.3f} "
                  f"Hmsg {row.get('car0_ent_msg', 0):.2f} {time.time()-t0:.0f}s", flush=True)

    run = os.path.join(out_dir, f"b{bits}_s{seed}")
    os.makedirs(run, exist_ok=True)
    pd.DataFrame(log).to_csv(os.path.join(run, "train_log.csv"), index=False)
    torch.save({"bits": bits, "obs_dim": obs_dim, "hidden": tc["hidden"],
                "state_dicts": [p.state_dict() for p in pols]}, os.path.join(run, "policies.pt"))
    manifest = dict(config=cfg, bits=bits, seed=seed, env_steps=steps, wall_seconds=time.time() - t0,
                    python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                    machine=platform.machine(), cpu_threads=1)
    with open(os.path.join(run, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
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
    return pols, ck["bits"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/runs")
    ap.add_argument("--total_steps", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.total_steps:
        cfg["train"]["total_steps"] = a.total_steps
    train(cfg, a.bits, a.seed, a.out)
