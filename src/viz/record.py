"""Record traffic-junction episodes for replay (HTML viewer, GIF).

An episode record is plain JSON:
  frames[t] = state at step t (before the step-t action), with the move and symbol each car chose at t,
              plus the events produced by step t-1 -> t (collision cells, arrivals, spawns).
"""
from __future__ import annotations

import numpy as np
import torch

from src.agents.scripted_tj import SCRIPTED

ARRIVAL_NAMES = {"easy": ["top", "left"], "medium": ["top", "bottom", "left", "right"]}
TURN_NAMES = ["straight", "right", "left"]


def route_labels(env):
    labels = []
    for k, a in enumerate(env.path_arrival):
        if env.difficulty == "easy":
            labels.append(["top → bottom", "left → right"][k])
        elif env.difficulty == "medium":
            j = env.paths_by_arrival[a].index(k)
            labels.append(f"from {ARRIVAL_NAMES['medium'][a]}, {TURN_NAMES[j]}")
        else:
            j = env.paths_by_arrival[a].index(k)
            labels.append(f"entry {a + 1}, route {j + 1}")
    return labels


def env_meta(env):
    return dict(difficulty=env.difficulty, H=env.H, W=env.W, road=env.road.tolist(),
                paths=[p.tolist() for p in env.paths], path_arrival=list(env.path_arrival),
                route_labels=route_labels(env), n_arrival=env.n_arrival, bits=env.bits,
                n_max=env.n_max, max_steps=env.max_steps, vision=env.vision, comm_radius=env.comm_radius,
                crash_mode=env.crash_mode)


class ScriptedDriver:
    def __init__(self, name, seed=0):
        self.name, self.fn, self.rng = name, SCRIPTED[name], np.random.default_rng(seed)

    def reset(self, env):
        pass

    def act(self, env, obs):
        acts = self.fn(env, self.rng)
        return acts, {}


class PopulationDriver:
    """Drives each arriving car with a network drawn at random from the population (as in training),
    or slot i with network i when the population is 'fixed'."""

    def __init__(self, pols, mode="random", seed=0, greedy=False):
        self.pols, self.mode, self.greedy = pols, mode, greedy
        self.rng = np.random.default_rng(seed)
        self.name = f"learned ({len(pols)} networks, {mode})"

    def reset(self, env):
        self.uid = np.full(env.n_max, -1)
        self.net = np.full(env.n_max, -1)

    def act(self, env, obs):
        acts, nets = {}, {}
        for i, a in enumerate(env.possible_agents):
            if not env.active[i]:
                continue
            if self.uid[i] != env.uid[i]:                  # a new car arrived in this slot
                self.uid[i] = env.uid[i]
                self.net[i] = i if self.mode == "fixed" else int(self.rng.integers(len(self.pols)))
            with torch.no_grad():
                mv, mg, *_ = self.pols[self.net[i]].act(torch.as_tensor(obs[a])[None], greedy=self.greedy)
            mv, mg = int(mv), int(mg)
            acts[a] = mv if env.bits == 0 else np.array([mv, mg])
            nets[i] = int(self.net[i])
        return acts, nets


def record_episode(env, driver, seed, add_rate=None):
    opts = {"add_rate": add_rate} if add_rate is not None else None
    obs, _ = env.reset(seed=seed, options=opts)
    driver.reset(env)
    frames, hit, spawned, arrived = [], [], [], []
    total = dict(collisions=0, arrived=0, spawned=0)
    while True:
        done = not env.agents
        acts, nets = ({}, {}) if done else driver.act(env, obs)
        cars = []
        for c in env.snapshot():
            a = env.possible_agents[c["slot"]]
            act = np.asarray(acts.get(a, [1, 0])).reshape(-1)
            cars.append(dict(s=c["slot"], u=c["uid"], r=c["pos"][0], c=c["pos"][1], p=c["path"], q=c["prog"],
                             tau=c["tau"], x=int(c["crashed"]), m=None if done else int(act[0]),
                             g=None if (done or env.bits == 0) else int(act[1]), k=nets.get(c["slot"])))
        frames.append(dict(t=env.t, cars=cars, hit=hit, spawned=spawned, arrived=arrived, total=dict(total)))
        if done:
            break
        before = {i: env.pos(i) for i in np.flatnonzero(env.active)}
        obs, rew, te, tr, info = env.step(acts)
        ev = env.events[-1]
        hit = sorted({env.pos(i) for i in ev["collisions"] if env.active[i]} |
                     {before[i] for i in ev["collisions"] if not env.active[i] and i in before})
        hit = [list(h) for h in hit]
        spawned, arrived = ev["spawned"], ev["arrived"]
        total["collisions"] += len(ev["collisions"])
        total["arrived"] += len(arrived)
        total["spawned"] += len(spawned)
    st = env.stats
    return dict(seed=int(seed), add_rate=env.add_rate, policy=driver.name, frames=frames,
                summary=dict(success=1 - st["failed"], spawned=st["spawned"], completed=st["completed"],
                             cars_collided=st["cars_collided"], collision_steps=st["collision_events"]))
