import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from src.envs.intersection import BRAKE, GAS, IntersectionEnv, build_routes


def act(env, moves, msgs=None):
    msgs = msgs or [0] * env.n
    if env.bits == 0:
        return {a: moves[i] for i, a in enumerate(env.possible_agents) if a in env.agents}
    return {a: np.array([moves[i], msgs[i]]) for i, a in enumerate(env.possible_agents) if a in env.agents}


def msg_slice(env):
    ns = env.n_sym * (env.n - 1)
    return slice(env._obs_dim - ns, env._obs_dim)


@pytest.mark.parametrize("bits", [0, 1, 2, 4])
def test_parallel_api(bits):
    parallel_api_test(IntersectionEnv(bits=bits), num_cycles=300)


def test_paths_are_contiguous_and_unique():
    for (a, r), p in build_routes(8).items():
        steps = np.abs(np.diff(p, axis=0)).sum(1)
        assert (steps == 1).all(), (a, r)
        assert len({tuple(x) for x in p}) == len(p)


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_alphabet_size(bits):
    env = IntersectionEnv(bits=bits)
    assert env.action_space("car_0").nvec.tolist() == [2, 2 ** bits]


def test_zero_bit_has_no_message_features():
    e0, e1 = IntersectionEnv(bits=0), IntersectionEnv(bits=1)
    assert e1._obs_dim - e0._obs_dim == 2
    assert e0.action_space("car_0").n == 2


def test_message_one_tick_delay():
    env = IntersectionEnv(bits=2)
    obs, _ = env.reset(options={"scenario": ([0, 2], [0, 0], [0, 0])})
    sl = msg_slice(env)
    assert obs["car_1"][sl].argmax() == 0                 # nothing sent yet: neutral symbol
    obs, *_ = env.step(act(env, [BRAKE, BRAKE], [3, 0]))  # car_0 sends 3 at t=0
    assert obs["car_1"][sl].argmax() == 3                 # seen at t=1, not earlier
    assert obs["car_0"][sl].argmax() == 0
    obs, *_ = env.step(act(env, [BRAKE, BRAKE], [1, 2]))
    assert obs["car_1"][sl].argmax() == 1 and obs["car_0"][sl].argmax() == 2


def test_collision_penalizes_only_involved_cars():
    env = IntersectionEnv(n_agents=3, bits=0)
    # W straight and N straight meet at (4,3); E right turn never enters that cell
    env.reset(options={"scenario": ([0, 1, 2], [0, 0, 1], [0, 0, 0])})
    for _ in range(3):
        env.step(act(env, [GAS, GAS, BRAKE]))
    # W car at (4,3), N car at (3,3); car_2 waits on its own lane
    assert env.pos(0) == (4, 3) and env.pos(1) == (3, 3) and not env.crashed.any()
    _, rew, term, _, info = env.step(act(env, [BRAKE, GAS, BRAKE]))  # N enters (4,3) where W sits
    assert env.crashed[0] and env.crashed[1] and not env.crashed[2]
    assert rew["car_0"] == pytest.approx(env.R_time + env.R_col)
    assert rew["car_1"] == pytest.approx(env.R_time + env.R_col)
    assert rew["car_2"] == pytest.approx(env.R_time)


def test_collision_is_order_independent():
    sc = ([0, 1], [0, 0], [0, 1])
    outs = []
    for order in (["car_0", "car_1"], ["car_1", "car_0"]):
        env = IntersectionEnv(bits=0)
        env.reset(options={"scenario": sc})
        while env.agents:
            a = {k: GAS for k in order if k in env.agents}
            env.step(a)
        outs.append((env.crashed.tolist(), env.arrived.tolist(), env.ticks.tolist()))
    assert outs[0] == outs[1]


def test_inactive_car_gets_no_reward():
    env = IntersectionEnv(bits=0)
    env.reset(options={"scenario": ([0, 2], [1, 0], [0, 0])})   # W right turn, E straight: disjoint
    total = {a: 0.0 for a in env.possible_agents}
    seen_after_done = False
    done = set()
    while env.agents:
        _, rew, te, tr, _ = env.step({a: GAS if a == "car_0" else BRAKE for a in env.agents})
        for a, r in rew.items():
            if a in done:
                seen_after_done = True
            total[a] += r
            if te[a] or tr[a]:
                done.add(a)
    assert not seen_after_done
    assert env.arrived[0]
    assert total["car_0"] == pytest.approx(env.R_arr + env.R_time * len(env.path[0]))


def test_private_route_not_in_other_obs():
    a, b = IntersectionEnv(bits=1), IntersectionEnv(bits=1)
    oa, _ = a.reset(options={"scenario": ([0, 1], [0, 0], [2, 2])})
    ob, _ = b.reset(options={"scenario": ([0, 1], [0, 2], [2, 2])})   # only car_1's route differs
    assert np.array_equal(oa["car_0"], ob["car_0"])
    assert not np.array_equal(oa["car_1"], ob["car_1"])


def test_reset_determinism():
    e1, e2 = IntersectionEnv(bits=1), IntersectionEnv(bits=1)
    o1, _ = e1.reset(seed=7)
    o2, _ = e2.reset(seed=7)
    for k in o1:
        assert np.array_equal(o1[k], o2[k])
