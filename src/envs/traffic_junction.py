"""Traffic Junction (TJ) with a b-bit discrete channel and individual rewards.

Follows the standard TJ task of CommNet (Sukhbaatar et al., 2016), IC3Net (Singh et al., 2019)
and TarMAC (Das et al., 2019); routes are generated with IC3Net's MIT-licensed helper
(`tj_routes.py`). Standard elements kept as in IC3Net's implementation:

  * levels easy / medium / hard (grid 7 / 14 / 18, N_max 5 / 10 / 20, 20 / 40 / 60 steps),
  * cars arrive over time: at every step, each arrival point adds a car with probability
    `add_rate` while fewer than N_max cars are on the grid; each car gets a random route,
  * actions GAS (advance one cell) / BRAKE (stay); a car leaves when it drives past its last cell,
  * reward per car per step: -0.01 * tau (tau = steps since the car entered) and -10 for every
    step in which it shares a cell with another car; no arrival bonus,
  * crashed cars keep driving (crash_mode="continue"); an episode with any collision is a failure.

Changes for this study (all documented in the methodology):

  * every car trains on its own reward (alpha = 0); alpha > 0 mixes in the mean reward of all
    cars on the grid this step, itself included (alpha = 1: shared team reward),
  * an explicit channel: each car emits one of 2**b symbols per step, delivered one step later;
    a car receives the MEAN of the one-hot symbols of all other senders (CommNet/IC3Net-style
    averaging), optionally only from senders within `comm_radius` cells (CommNet's local variant),
  * two cars that swap cells in one step also collide (`swap_collision`),
  * crash_mode="remove" (pilot behaviour) is available as an ablation.

PettingZoo Parallel API. Agents are car *slots* "car_0".."car_{N_max-1}". A slot is empty until a
car arrives in it; empty slots receive an all-zero observation and reward 0 and their actions are
ignored (as with IC3Net's alive mask). A slot can host several cars, one after another, in an
episode; `infos[slot]["car_done"]` marks the step a car leaves.
"""
from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src.envs.tj_routes import get_road_blocks, get_routes

GAS, BRAKE = 0, 1

PRESETS = {
    #           dim  n_max  steps  add_rate (start, end of curriculum)
    "easy":   dict(dim=6, n_max=5, max_steps=20, add_rate=(0.1, 0.3)),
    "medium": dict(dim=14, n_max=10, max_steps=40, add_rate=(0.05, 0.2)),
    "hard":   dict(dim=18, n_max=20, max_steps=60, add_rate=(0.02, 0.05)),
}


def build_layout(difficulty: str, dim: int):
    """Return (road grid HxW of 0/1, list of paths, list of arrival index per path)."""
    if difficulty == "easy":
        h = w = dim + 1                       # IC3Net makes the easy grid odd-sized
        road = np.zeros((h, w), int)
        road[h // 2, :] = 1
        road[:, w // 2] = 1
        paths = [np.array([(i, w // 2) for i in range(h)]),      # top -> bottom
                 np.array([(h // 2, i) for i in range(w)])]       # left -> right
        return road, paths, [0, 1]
    h = w = dim
    road = np.zeros((h, w), int)
    for blk in get_road_blocks(w, h, difficulty):
        road[blk] = 1
    routes = get_routes((h, w), road, difficulty)
    paths, arrival = [], []
    for a, rs in enumerate(routes):
        for p in rs:
            paths.append(np.array(p))
            arrival.append(a)
    return road, paths, arrival


class TrafficJunctionEnv(ParallelEnv):
    metadata = {"name": "traffic_junction_comm_v1", "render_modes": ["ansi"]}

    def __init__(self, difficulty="medium", n_max=None, dim=None, max_steps=None, vision=0,
                 bits=0, comm_radius=None, crash_mode="continue", swap_collision=True,
                 reward_time=-0.01, reward_collision=-10.0, alpha=0.0, add_rate=None,
                 render_mode=None):
        assert difficulty in PRESETS
        assert crash_mode in ("continue", "remove")
        p = PRESETS[difficulty]
        self.difficulty = difficulty
        self.dim = dim or p["dim"]
        self.n_max = n_max or p["n_max"]
        self.max_steps = max_steps or p["max_steps"]
        self.vision, self.bits, self.comm_radius = vision, bits, comm_radius
        self.n_sym = 2 ** bits if bits > 0 else 0
        self.crash_mode, self.swap_collision = crash_mode, swap_collision
        self.R_time, self.R_col, self.alpha = reward_time, reward_collision, alpha
        self.add_rate = add_rate if add_rate is not None else p["add_rate"][1]
        self.render_mode = render_mode

        self.road, self.paths, self.path_arrival = build_layout(difficulty, self.dim)
        self.H, self.W = self.road.shape
        self.npath = len(self.paths)
        self.n_arrival = max(self.path_arrival) + 1
        self.paths_by_arrival = [[k for k, a in enumerate(self.path_arrival) if a == ai]
                                 for ai in range(self.n_arrival)]
        self.possible_agents = [f"car_{i}" for i in range(self.n_max)]
        self.rng = np.random.default_rng()
        self.rng_eval = np.random.default_rng(0)   # used only by evaluation interventions
        self.inbox_transform = None
        v = 2 * vision + 1
        self.obs_layout = {"route": self.npath, "pos": 2, "progress": 1, "tau": 1, "last_move": 1,
                           "time": 1, "vision": v * v, "inbox": self.n_sym}
        self._obs_dim = sum(self.obs_layout.values())

    # ---------------- spaces ----------------
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(0.0, 1.0, shape=(self._obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(2) if self.bits == 0 else spaces.MultiDiscrete([2, self.n_sym])

    def inbox_slice(self):
        return slice(self._obs_dim - self.n_sym, self._obs_dim)

    # ---------------- core ----------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options and "add_rate" in options:
            self.add_rate = float(options["add_rate"])
        n = self.n_max
        self.agents = list(self.possible_agents)
        self.t = 0
        self.active = np.zeros(n, bool)
        self.path_id = np.full(n, -1)
        self.prog = np.zeros(n, int)
        self.tau = np.zeros(n, int)
        self.last_act = np.zeros(n, int)
        self.uid = np.full(n, -1)
        self.car_crashed = np.zeros(n, bool)     # current car has collided at least once
        self.next_uid = 0
        self.senders = []                        # (slot, (r, c), symbol) sent last step
        self.stats = dict(spawned=0, completed=0, collision_events=0, cars_collided=0, failed=0)
        self.events = []                         # per-step event log for the recorder
        self.exit_tau = []
        obs = {a: self._obs(i) for i, a in enumerate(self.possible_agents)}
        return obs, {a: {"active": False} for a in self.possible_agents}

    def pos(self, i):
        return tuple(int(x) for x in self.paths[self.path_id[i]][self.prog[i]])

    def _spawn(self, slot, path):
        self.active[slot] = True
        self.path_id[slot] = path
        self.prog[slot] = 0
        self.tau[slot] = 0
        self.last_act[slot] = BRAKE
        self.uid[slot] = self.next_uid
        self.next_uid += 1
        self.car_crashed[slot] = False
        self.stats["spawned"] += 1

    def step(self, actions):
        n = self.n_max
        moves = np.full(n, BRAKE)
        msgs = np.zeros(n, int)
        for i, a in enumerate(self.possible_agents):
            if self.active[i] and a in actions:
                act = np.asarray(actions[a]).reshape(-1)
                moves[i] = int(act[0])
                if self.bits > 0:
                    msgs[i] = int(act[1])
        rew = np.zeros(n)
        was_active = self.active.copy()          # cars that act this step
        got_reward = self.active.copy()          # cars that get a reward entry this step
        car_done = np.zeros(n, bool)
        arrived = np.zeros(n, bool)
        old_pos = {i: self.pos(i) for i in range(n) if self.active[i]}
        # messages are sent from the sender's position at the start of the step
        new_senders = [(i, old_pos[i], int(msgs[i])) for i in range(n) if self.active[i]] if self.n_sym else []

        # 1) move every car on the grid at once
        for i in range(n):
            if not self.active[i]:
                continue
            self.tau[i] += 1
            self.last_act[i] = moves[i]
            if moves[i] == GAS:
                if self.prog[i] + 1 >= len(self.paths[self.path_id[i]]):
                    self.active[i] = False           # drives off the grid: done, no penalty this step
                    car_done[i] = arrived[i] = True
                    self.stats["completed"] += 1
                    continue
                self.prog[i] += 1
            rew[i] += self.R_time * self.tau[i]

        # 2) new arrivals (after the moves, as in IC3Net)
        spawned = np.zeros(n, bool)
        for ai in range(self.n_arrival):
            if self.active.sum() >= n:
                break
            if self.rng.uniform() <= self.add_rate:
                free = np.flatnonzero(~self.active)
                slot = int(self.rng.choice(free))
                self._spawn(slot, int(self.rng.choice(self.paths_by_arrival[ai])))
                spawned[slot] = True
                got_reward[slot] = True

        # 3) collisions among cars on the grid
        on = np.flatnonzero(self.active)
        hit = set()
        for x in range(len(on)):
            for y in range(x + 1, len(on)):
                i, j = on[x], on[y]
                pi, pj = self.pos(i), self.pos(j)
                swap = (self.swap_collision and i in old_pos and j in old_pos
                        and pi == old_pos[j] and pj == old_pos[i])
                if pi == pj or swap:
                    hit.update((int(i), int(j)))
        if hit:
            self.stats["collision_events"] += 1
            self.stats["failed"] = 1
        for i in hit:
            rew[i] += self.R_col
            if not self.car_crashed[i]:
                self.stats["cars_collided"] += 1
            self.car_crashed[i] = True
            if self.crash_mode == "remove":
                self.active[i] = False
                car_done[i] = True

        # 4) prosociality mixing over the cars that received a reward this step
        if self.alpha > 0 and got_reward.sum() > 1:
            idx = np.flatnonzero(got_reward)
            raw = rew.copy()
            team = raw[idx].mean()               # mean over all cars that got a reward this step, self included
            for i in idx:
                rew[i] = (1 - self.alpha) * raw[i] + self.alpha * team

        self.senders = new_senders
        self.t += 1
        self.exit_tau = [int(self.tau[i]) for i in np.flatnonzero(arrived)]
        self.events.append(dict(t=self.t, collisions=sorted(hit), spawned=np.flatnonzero(spawned).tolist(),
                                arrived=np.flatnonzero(arrived).tolist()))

        done_all = self.t >= self.max_steps
        obs, rews, terms, truncs, infos = {}, {}, {}, {}, {}
        for i, a in enumerate(self.possible_agents):
            obs[a] = self._obs(i)
            r = float(rew[i]) if got_reward[i] else 0.0
            # A slot's previous car may leave (reward 0) and a new car arrive in the same step;
            # r_acted belongs to the car that acted, r_spawn to the car that just arrived.
            r_spawn = r if spawned[i] else 0.0
            r_acted = r - r_spawn if was_active[i] else 0.0
            rews[a] = r
            terms[a] = False
            truncs[a] = bool(done_all)
            infos[a] = {"active": bool(self.active[i]), "acted": bool(was_active[i]),
                        "spawned": bool(spawned[i]), "car_done": bool(car_done[i]), "arrived": bool(arrived[i]),
                        "collided": i in hit, "uid": int(self.uid[i]), "msg_sent": int(msgs[i]),
                        "r_acted": r_acted, "r_spawn": r_spawn}
        if done_all:
            self.agents = []
        return obs, rews, terms, truncs, infos

    def _obs(self, i):
        o = np.zeros(self._obs_dim, np.float32)
        if not self.active[i]:
            return o
        k = 0
        o[k + self.path_id[i]] = 1; k += self.npath
        r, c = self.pos(i)
        o[k] = r / (self.H - 1); o[k + 1] = c / (self.W - 1); k += 2
        o[k] = self.prog[i] / (len(self.paths[self.path_id[i]]) - 1); k += 1
        o[k] = min(self.tau[i] / self.max_steps, 1.0); k += 1
        o[k] = self.last_act[i]; k += 1
        o[k] = self.t / self.max_steps; k += 1
        v = self.vision
        for j in np.flatnonzero(self.active):
            if j == i:
                continue
            rj, cj = self.pos(j)
            if abs(rj - r) <= v and abs(cj - c) <= v:
                o[k + (rj - r + v) * (2 * v + 1) + (cj - c + v)] = 1
        k += (2 * v + 1) ** 2
        if self.n_sym:
            cnt = 0
            for j, (sr, sc), sym in self.senders:
                if j == i:
                    continue
                if self.comm_radius is not None and max(abs(sr - r), abs(sc - c)) > self.comm_radius:
                    continue
                if self.inbox_transform is not None:
                    sym = self.inbox_transform(i, j, sym, self.rng_eval)
                o[k + sym] += 1
                cnt += 1
            if cnt:
                o[k:k + self.n_sym] /= cnt
        return o

    # ---------------- helpers ----------------
    def snapshot(self):
        """State of every car on the grid (for the recorder)."""
        return [dict(slot=int(i), uid=int(self.uid[i]), pos=self.pos(i), path=int(self.path_id[i]),
                     prog=int(self.prog[i]), tau=int(self.tau[i]), crashed=bool(self.car_crashed[i]))
                for i in np.flatnonzero(self.active)]

    def render(self):
        g = [["#" if not self.road[r, c] else "." for c in range(self.W)] for r in range(self.H)]
        for i in np.flatnonzero(self.active):
            r, c = self.pos(i)
            g[r][c] = "X" if g[r][c] not in ("#", ".") else str(i % 10)
        return "\n".join("".join(row) for row in g)

    def state(self):
        raise NotImplementedError

    def close(self):
        pass
