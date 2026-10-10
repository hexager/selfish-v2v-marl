"""Scripted reference controllers. They read env internals; each says what it is allowed to know."""
import numpy as np

from src.envs.intersection import BRAKE, GAS


def _pack(env, moves):
    if env.bits == 0:
        return {a: int(moves[i]) for i, a in enumerate(env.possible_agents) if a in env.agents}
    return {a: np.array([moves[i], 0]) for i, a in enumerate(env.possible_agents) if a in env.agents}


def random_policy(env, rng):
    """Floor: uniform GAS/BRAKE."""
    return _pack(env, rng.integers(0, 2, size=env.n))


def always_gas(env, rng=None):
    """Reckless: never yields."""
    return _pack(env, np.zeros(env.n, int))


def _next_cell(env, i):
    if env.prog[i] + 1 >= len(env.path[i]):
        return None                       # exits the grid
    return tuple(env.path[i][env.prog[i] + 1])


def _in_box(env, cell):
    lo, hi = env.grid // 2 - 1, env.grid // 2
    return lo <= cell[0] <= hi and lo <= cell[1] <= hi


def local_right_of_way(env, rng=None):
    """No communication. Uses only what the car could see in its vision window:
    brake if the next cell holds a car, or if any car is visible and that car's
    approach has priority (lower approach index = higher priority) and this car is
    about to enter the junction box."""
    moves = np.zeros(env.n, int)
    for i in range(env.n):
        if not env.active[i]:
            continue
        nxt = _next_cell(env, i)
        r, c = env.pos(i)
        for j in range(env.n):
            if j == i or not env.active[j]:
                continue
            rj, cj = env.pos(j)
            if max(abs(rj - r), abs(cj - c)) > env.vision:
                continue
            entering = nxt is not None and _in_box(env, nxt) and not _in_box(env, (r, c))
            if nxt == (rj, cj) or (entering and env.approach[j] < env.approach[i]):
                moves[i] = BRAKE
    return _pack(env, moves)


def oracle(env, rng=None):
    """Upper bound with full information (all routes and positions): cell reservation
    in fixed priority order. Not comparable in information access."""
    moves = np.full(env.n, BRAKE)
    order = [i for i in np.argsort(env.approach) if env.active[i]]
    reserved = {env.pos(i) for i in order}
    for i in order:
        nxt = _next_cell(env, i)
        cur = env.pos(i)
        if nxt is None:
            moves[i] = GAS
            reserved.discard(cur)
            continue
        if nxt not in reserved:
            moves[i] = GAS
            reserved.discard(cur)
            reserved.add(nxt)
    return _pack(env, moves)


SCRIPTED = {"random": random_policy, "always_gas": always_gas,
            "local_right_of_way": local_right_of_way, "oracle": oracle}
