"""Scripted reference drivers for the traffic-junction env. Each returns {agent: action}."""
import numpy as np

from src.envs.traffic_junction import BRAKE, GAS


def _pack(env, moves):
    out = {}
    for i, a in enumerate(env.possible_agents):
        if not env.active[i]:
            continue
        out[a] = int(moves[i]) if env.bits == 0 else np.array([moves[i], 0])
    return out


def random_policy(env, rng):
    """Floor: GAS or BRAKE uniformly at random."""
    return _pack(env, rng.integers(0, 2, size=env.n_max))


def always_gas(env, rng=None):
    """Never yields."""
    return _pack(env, np.zeros(env.n_max, int))


def _next_cell(env, i):
    p = env.paths[env.path_id[i]]
    return None if env.prog[i] + 1 >= len(p) else tuple(int(x) for x in p[env.prog[i] + 1])


def oracle(env, rng=None):
    """Upper bound with full information: cell reservation, oldest car first (largest tau).
    It cannot prevent a new car from arriving on an occupied entry cell (a property of TJ)."""
    moves = np.full(env.n_max, BRAKE)
    on = np.flatnonzero(env.active)
    order = sorted(on, key=lambda i: (-env.tau[i], i))
    reserved = {env.pos(i) for i in on}
    for i in order:
        nxt = _next_cell(env, i)
        if nxt is None:
            moves[i] = GAS
            reserved.discard(env.pos(i))
        elif nxt not in reserved:
            moves[i] = GAS
            reserved.discard(env.pos(i))
            reserved.add(nxt)
    return _pack(env, moves)


SCRIPTED = {"random": random_policy, "always_gas": always_gas, "oracle": oracle}


def junction_mask(env):
    rows = env.road.sum(1) > env.W / 2          # rows that carry a horizontal road
    cols = env.road.sum(0) > env.H / 2          # columns that carry a vertical road
    return rows[:, None] & cols[None, :] & (env.road > 0)


def honest_signal(env, rng=None):
    """1-bit protocol with the same channel the learners get (mean of others' symbols, one step late).
    Send 1 when in the junction or entering it this step; a car about to enter waits if any car signalled 1.
    Shows how much a single honest bit can be worth under this channel design."""
    J = junction_mask(env)
    sl = env.inbox_slice()
    moves = np.zeros(env.n_max, int)
    msgs = np.zeros(env.n_max, int)
    for i in np.flatnonzero(env.active):
        r, c = env.pos(i)
        nxt = _next_cell(env, i)
        entering = nxt is not None and J[nxt] and not J[r, c]
        heard_busy = env._obs(i)[sl][1] > 0 if env.n_sym else False
        moves[i] = BRAKE if (entering and heard_busy) else GAS
        msgs[i] = int(J[r, c] or (entering and moves[i] == GAS))
    out = {}
    for i, a in enumerate(env.possible_agents):
        if env.active[i]:
            out[a] = np.array([moves[i], msgs[i]])
    return out


SCRIPTED["honest_signal"] = honest_signal


def _ahead_blocked(env, i):
    """Car i sees (vision >= 1) that the next cell on its route holds a car."""
    nxt = _next_cell(env, i)
    if nxt is None or env.vision < 1:
        return False
    return any(env.pos(j) == nxt for j in np.flatnonzero(env.active) if j != i)


def follow_gap(env, rng=None):
    """No communication: GAS unless the car ahead is visible in the next cell (needs vision >= 1)."""
    moves = np.array([BRAKE if env.active[i] and _ahead_blocked(env, i) else GAS for i in range(env.n_max)])
    return _pack(env, moves)


def honest_signal_gap(env, rng=None):
    """follow_gap plus the 1-bit junction protocol of honest_signal."""
    J = junction_mask(env)
    sl = env.inbox_slice()
    out = {}
    for i, a in enumerate(env.possible_agents):
        if not env.active[i]:
            continue
        r, c = env.pos(i)
        nxt = _next_cell(env, i)
        entering = nxt is not None and J[nxt] and not J[r, c]
        busy = env._obs(i)[sl][1] > 0 if env.n_sym else False
        move = BRAKE if (_ahead_blocked(env, i) or (entering and busy)) else GAS
        msg = int(J[r, c] or (entering and move == GAS))
        out[a] = np.array([move, msg]) if env.bits else move
    return out


SCRIPTED["follow_gap"] = follow_gap
SCRIPTED["honest_signal_gap"] = honest_signal_gap
