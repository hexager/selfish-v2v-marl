"""Evaluation on the full enumerated scenario bank, with message interventions and cross-play.

Interventions (applied only to the receiver's observation; physics and sender unchanged):
  none     - normal play
  mute     - every received symbol replaced by symbol 0 (constant)
  random   - every received symbol replaced by a uniform random symbol
  permute  - cyclic shift of the alphabet (s -> s+1 mod 2^b) applied to received symbols
"""
from __future__ import annotations

import itertools

import numpy as np
import torch

from src.agents.scripted import SCRIPTED
from src.envs.intersection import IntersectionEnv


def scenario_bank(n_agents=2, max_offset=2):
    """Every (approaches, routes, start offsets) tuple. For 2 cars: 12*9*9 = 972 scenarios."""
    bank = []
    for appr in itertools.permutations(range(4), n_agents):
        for rts in itertools.product(range(3), repeat=n_agents):
            for offs in itertools.product(range(max_offset + 1), repeat=n_agents):
                bank.append((list(appr), list(rts), list(offs)))
    return bank


def _intervene(o, env, mode, rng, perm):
    if mode == "none" or env.n_sym == 0:
        return o
    o = o.copy()
    ns, n_in = env.n_sym, env.n - 1
    base = len(o) - ns * n_in
    for s in range(n_in):
        sl = slice(base + s * ns, base + (s + 1) * ns)
        sym = int(np.argmax(o[sl]))
        new = {"mute": 0, "random": int(rng.integers(ns)), "permute": int(perm[sym])}[mode]
        o[sl] = 0
        o[sl.start + new] = 1
    return o


def evaluate(env_kwargs, policies=None, scripted=None, reps=2, intervention="none", seed=12345,
             greedy=False, record_messages=False):
    """policies: list of CarPolicy (one per car) or None with scripted=name."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = IntersectionEnv(**env_kwargs)
    perm = (np.arange(max(env.n_sym, 1)) + 1) % max(env.n_sym, 1)   # cyclic shift: never the identity
    bank = scenario_bank(env.n, env.max_start_offset)
    rows, msgs = [], []
    for k, sc in enumerate(bank):
        for r in range(reps):
            obs, _ = env.reset(seed=int(rng.integers(2**31)), options={"scenario": sc})
            ret = np.zeros(env.n)
            while env.agents:
                if scripted:
                    acts = SCRIPTED[scripted](env, rng)
                else:
                    acts = {}
                    for i, a in enumerate(env.possible_agents):
                        if a not in env.agents:
                            continue
                        x = torch.as_tensor(_intervene(obs[a], env, intervention, rng, perm))[None]
                        with torch.no_grad():
                            mv, mg, *_ = policies[i].act(x, greedy=greedy)
                        mv, mg = int(mv), int(mg)
                        acts[a] = mv if env.bits == 0 else np.array([mv, mg])
                        if record_messages and env.bits > 0:
                            msgs.append(dict(scenario=k, t=env.t, car=i, route=int(env.route[i]),
                                             approach=int(env.approach[i]), prog=int(env.prog[i]),
                                             move=mv, msg=mg))
                obs, rew, te, tr, info = env.step(acts)
                for i, a in enumerate(env.possible_agents):
                    if a in rew:
                        ret[i] += rew[a]
            rows.append(dict(scenario=k, rep=r, crashed=env.crashed.copy(), arrived=env.arrived.copy(),
                             ticks=env.ticks.copy(), ret=ret))
    return summarize(rows), rows, msgs


def summarize(rows):
    crashed = np.array([r["crashed"] for r in rows])
    arrived = np.array([r["arrived"] for r in rows])
    ticks = np.array([r["ticks"] for r in rows])
    ret = np.array([r["ret"] for r in rows])
    return dict(
        episodes=len(rows),
        collision_ep_rate=float(crashed.any(1).mean()),
        crash_rate_per_car=float(crashed.mean()),
        arrival_rate=float(arrived.mean()),
        timeout_rate=float((~arrived & ~crashed).mean()),
        mean_ticks_arrived=float(ticks[arrived].mean()) if arrived.any() else float("nan"),
        mean_return=float(ret.mean()),
        welfare=float(ret.sum(1).mean()),               # evaluation-only social welfare
        worst_car_return=float(ret.min(1).mean()),
        **{f"car{i}_return": float(ret[:, i].mean()) for i in range(ret.shape[1])},
    )


def mutual_information(x, y):
    """Plug-in MI estimate in bits between two discrete sequences."""
    x, y = np.asarray(x), np.asarray(y)
    mi = 0.0
    for a in np.unique(x):
        pa = (x == a).mean()
        for b in np.unique(y):
            pab = ((x == a) & (y == b)).mean()
            if pab > 0:
                mi += pab * np.log2(pab / (pa * (y == b).mean()))
    return mi
