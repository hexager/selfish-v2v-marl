"""Two-lane four-way intersection with private routes and a b-bit discrete channel.

Geometry and incentives follow the traffic-junction task of CommNet / IC3Net /
TarMAC (GAS/BRAKE actions, private route id, small vision window, per-car
collision penalty), with three changes for this study:
  * every car optimizes only its own reward (no reward mixing),
  * an explicit discrete message channel with alphabet size 2**b,
  * messages are delivered with a one-tick delay (sent at t, seen at t+1).

PettingZoo Parallel API. Agents: "car_0" .. "car_{n-1}".
"""
from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

GAS, BRAKE = 0, 1
APPROACHES = ["W", "N", "E", "S"]          # entry side
ROUTES = ["straight", "right", "left"]


def _west_routes(n: int) -> list[list[tuple[int, int]]]:
    """Routes for a car entering from the west, eastbound in row n//2 (right-hand traffic)."""
    lo, hi = n // 2 - 1, n // 2           # junction rows/cols
    straight = [(hi, c) for c in range(n)]
    right = [(hi, c) for c in range(lo + 1)] + [(r, lo) for r in range(hi + 1, n)]
    left = [(hi, c) for c in range(hi + 1)] + [(r, hi) for r in range(lo, -1, -1)]
    return [straight, right, left]


def _rot_cw(cell, n):
    r, c = cell
    return (c, n - 1 - r)


def build_routes(n: int) -> dict[tuple[int, int], np.ndarray]:
    """(approach_idx, route_idx) -> array of cells. W->N->E->S is a clockwise rotation."""
    out = {}
    base = _west_routes(n)
    for a in range(4):
        for k, path in enumerate(base):
            p = path
            for _ in range(a):
                p = [_rot_cw(x, n) for x in p]
            out[(a, k)] = np.array(p, dtype=np.int64)
    return out


class IntersectionEnv(ParallelEnv):
    metadata = {"name": "selfish_intersection_v0", "render_modes": ["ansi"]}

    def __init__(self, n_agents=2, grid=8, horizon=20, vision=1, bits=1,
                 max_start_offset=2, reward_arrival=1.0, reward_collision=-2.0,
                 reward_time=-0.05, reward_mode="individual", render_mode=None):
        assert n_agents >= 2 and n_agents <= 4
        assert reward_mode in ("individual", "team")   # "team" is a diagnostic only
        self.n, self.grid, self.horizon, self.vision = n_agents, grid, horizon, vision
        self.bits = bits
        self.n_sym = 2 ** bits if bits > 0 else 0
        self.max_start_offset = max_start_offset
        self.R_arr, self.R_col, self.R_time = reward_arrival, reward_collision, reward_time
        self.reward_mode = reward_mode
        self.render_mode = render_mode
        self.possible_agents = [f"car_{i}" for i in range(n_agents)]
        self.routes = build_routes(grid)
        self.rng = np.random.default_rng()
        self._obs_dim = 4 + 3 + 2 + 1 + 1 + 1 + (2 * vision + 1) ** 2 + (n_agents - 1) * self.n_sym

    # ---------------- spaces ----------------
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(0.0, 1.0, shape=(self._obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        if self.bits == 0:
            return spaces.Discrete(2)
        return spaces.MultiDiscrete([2, self.n_sym])

    # ---------------- core ----------------
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self.t = 0
        appr = self.rng.choice(4, size=self.n, replace=False)
        rts = self.rng.integers(0, 3, size=self.n)
        offs = self.rng.integers(0, self.max_start_offset + 1, size=self.n)
        if options and "scenario" in options:      # explicit scenario for held-out banks / tests
            appr, rts, offs = (np.array(x) for x in options["scenario"])
        self.approach, self.route, self.prog = appr.astype(int), rts.astype(int), offs.astype(int)
        self.path = [self.routes[(int(a), int(r))] for a, r in zip(self.approach, self.route)]
        self.active = np.ones(self.n, bool)
        self.arrived = np.zeros(self.n, bool)
        self.crashed = np.zeros(self.n, bool)
        self.last_act = np.zeros(self.n, int)
        self.ticks = np.zeros(self.n, int)
        self.inbox = np.zeros(self.n, int)        # message each car sent last tick (0 = neutral)
        self.sent = np.zeros(self.n, int)
        obs = {a: self._obs(i) for i, a in enumerate(self.possible_agents)}
        infos = {a: {} for a in self.possible_agents}
        return obs, infos

    def pos(self, i):
        return tuple(self.path[i][self.prog[i]])

    def step(self, actions):
        moves = np.full(self.n, BRAKE)
        msgs = np.zeros(self.n, int)
        for i, a in enumerate(self.possible_agents):
            if a not in actions or not self.active[i]:
                continue
            act = actions[a]
            if self.bits == 0:
                moves[i] = int(np.asarray(act).reshape(-1)[0])
            else:
                act = np.asarray(act).reshape(-1)
                moves[i], msgs[i] = int(act[0]), int(act[1])
        rew = np.zeros(self.n)
        was_active = self.active.copy()
        old_pos = {i: self.pos(i) for i in range(self.n) if self.active[i]}

        # simultaneous movement
        for i in range(self.n):
            if not self.active[i]:
                continue
            self.ticks[i] += 1
            rew[i] += self.R_time
            self.last_act[i] = moves[i]
            if moves[i] == GAS:
                if self.prog[i] + 1 >= len(self.path[i]):
                    self.active[i] = False          # leaves the grid
                    self.arrived[i] = True
                    rew[i] += self.R_arr
                else:
                    self.prog[i] += 1

        # collisions: same cell after the move, or two cars swapping cells
        on_grid = [i for i in range(self.n) if self.active[i]]
        hit = set()
        for x in range(len(on_grid)):
            for y in range(x + 1, len(on_grid)):
                i, j = on_grid[x], on_grid[y]
                pi, pj = self.pos(i), self.pos(j)
                swap = pi == old_pos[j] and pj == old_pos[i]
                if pi == pj or swap:
                    hit.update((i, j))
        for i in hit:
            rew[i] += self.R_col
            self.active[i] = False
            self.crashed[i] = True

        # messages are buffered: sent now, visible next tick. Inactive senders emit 0.
        self.sent = np.where(was_active, msgs, 0)
        self.inbox = self.sent.copy()
        self.t += 1

        if self.reward_mode == "team":
            rew = np.full(self.n, rew.sum() / self.n)

        trunc_all = self.t >= self.horizon
        obs, rews, terms, truncs, infos = {}, {}, {}, {}, {}
        for i, a in enumerate(self.possible_agents):
            if a not in self.agents:
                continue
            obs[a] = self._obs(i)
            rews[a] = float(rew[i])
            terms[a] = bool(not self.active[i])
            truncs[a] = bool(trunc_all and self.active[i])
            infos[a] = {"arrived": bool(self.arrived[i]), "crashed": bool(self.crashed[i]),
                        "ticks": int(self.ticks[i]), "msg_sent": int(self.sent[i])}
        self.agents = [a for a in self.agents if not (terms[a] or truncs[a])]
        return obs, rews, terms, truncs, infos

    def _obs(self, i):
        o = np.zeros(self._obs_dim, np.float32)
        k = 0
        o[k + self.approach[i]] = 1; k += 4
        o[k + self.route[i]] = 1; k += 3
        if self.active[i]:
            r, c = self.pos(i)
            o[k] = r / (self.grid - 1); o[k + 1] = c / (self.grid - 1)
        k += 2
        o[k] = self.prog[i] / (len(self.path[i]) - 1); k += 1
        o[k] = self.last_act[i]; k += 1
        o[k] = self.t / self.horizon; k += 1
        v = self.vision
        if self.active[i]:
            r, c = self.pos(i)
            for j in range(self.n):
                if j != i and self.active[j]:
                    rj, cj = self.pos(j)
                    dr, dc = rj - r, cj - c
                    if abs(dr) <= v and abs(dc) <= v:
                        o[k + (dr + v) * (2 * v + 1) + (dc + v)] = 1
        k += (2 * v + 1) ** 2
        if self.n_sym:
            for j in range(self.n):
                if j == i:
                    continue
                o[k + self.inbox[j]] = 1
                k += self.n_sym
        return o

    def render(self):
        g = [["#"] * self.grid for _ in range(self.grid)]
        for r, c in {tuple(x) for p in self.routes.values() for x in p}:
            g[r][c] = "."
        for i in range(self.n):
            if self.active[i]:
                r, c = self.pos(i)
                g[r][c] = str(i)
        return "\n".join("".join(row) for row in g)

    def state(self):
        raise NotImplementedError

    def close(self):
        pass
