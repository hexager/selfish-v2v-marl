import numpy as np
import pytest
from pettingzoo.test import parallel_api_test

from src.envs.traffic_junction import BRAKE, GAS, TrafficJunctionEnv


def acts(env, moves, msgs=None):
    out = {}
    for i, a in enumerate(env.possible_agents):
        if env.active[i]:
            m = moves[i] if isinstance(moves, (list, np.ndarray)) else moves
            out[a] = int(m) if env.bits == 0 else np.array([m, 0 if msgs is None else msgs[i]])
    return out


def place(env, slot, path, prog=0):
    env._spawn(slot, path)
    env.prog[slot] = prog


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
@pytest.mark.parametrize("bits", [0, 1, 4])
def test_parallel_api(difficulty, bits):
    parallel_api_test(TrafficJunctionEnv(difficulty, bits=bits), num_cycles=200)


@pytest.mark.parametrize("difficulty,npath,grid", [("easy", 2, 7), ("medium", 12, 14), ("hard", 56, 18)])
def test_standard_layouts(difficulty, npath, grid):
    env = TrafficJunctionEnv(difficulty)
    assert env.npath == npath and env.H == grid
    for p in env.paths:
        assert (np.abs(np.diff(p, axis=0)).sum(1) == 1).all()       # one cell per step
        assert all(env.road[r, c] for r, c in p)                    # stays on the road
        assert len({tuple(x) for x in p}) == len(p)                 # never revisits a cell


def test_arrivals_respect_n_max_and_rate():
    env = TrafficJunctionEnv("medium", add_rate=1.0)
    env.reset(seed=0)
    for _ in range(10):
        env.step(acts(env, BRAKE))
        assert env.active.sum() <= env.n_max
    env = TrafficJunctionEnv("medium", add_rate=0.0)
    env.reset(seed=0)
    for _ in range(5):
        env.step({})
    assert env.active.sum() == 0


def test_time_penalty_grows_with_tau():
    env = TrafficJunctionEnv("medium", add_rate=0.0)
    env.reset(seed=0)
    place(env, 0, 0)
    rs = []
    for _ in range(3):
        _, r, *_ = env.step(acts(env, BRAKE))
        rs.append(r["car_0"])
    assert rs == pytest.approx([-0.01, -0.02, -0.03])


def test_collision_penalty_and_cars_continue():
    env = TrafficJunctionEnv("medium", add_rate=0.0)
    env.reset(seed=0)
    p = env.paths[0]
    place(env, 0, 0, prog=2)
    place(env, 1, 0, prog=1)                     # same route, one cell behind
    _, r, _, _, info = env.step(acts(env, [BRAKE, GAS] + [BRAKE] * 8))
    assert info["car_0"]["collided"] and info["car_1"]["collided"]
    assert r["car_0"] == pytest.approx(-10 - 0.01) and r["car_1"] == pytest.approx(-10 - 0.01)
    assert env.active[0] and env.active[1]       # TJ standard: crashed cars keep driving
    assert env.stats["failed"] == 1


def test_crash_mode_remove():
    env = TrafficJunctionEnv("medium", add_rate=0.0, crash_mode="remove")
    env.reset(seed=0)
    place(env, 0, 0, prog=2)
    place(env, 1, 0, prog=1)
    _, _, _, _, info = env.step(acts(env, [BRAKE, GAS] + [BRAKE] * 8))
    assert not env.active[0] and not env.active[1] and info["car_0"]["car_done"]


def test_exit_gives_no_penalty_and_frees_slot():
    env = TrafficJunctionEnv("medium", add_rate=0.0)
    env.reset(seed=0)
    place(env, 0, 0, prog=len(env.paths[0]) - 1)
    _, r, _, _, info = env.step(acts(env, GAS))
    assert info["car_0"]["arrived"] and info["car_0"]["car_done"] and r["car_0"] == 0.0
    assert not env.active[0]


def test_message_delay_and_mean_pooling():
    env = TrafficJunctionEnv("medium", bits=2, add_rate=0.0)
    env.reset(seed=0)
    place(env, 0, 0); place(env, 1, 3); place(env, 2, 6)
    sl = env.inbox_slice()
    o, *_ = env.step(acts(env, BRAKE, msgs=[1, 3, 3] + [0] * 7))
    assert np.allclose(o["car_0"][sl], [0, 0, 0, 1])                  # hears cars 1, 2 (both symbol 3)
    assert np.allclose(o["car_1"][sl], [0, 0.5, 0, 0.5])              # hears car 0 (1) and car 2 (3)
    o, *_ = env.step(acts(env, BRAKE, msgs=[2, 2, 2] + [0] * 7))
    assert np.allclose(o["car_0"][sl], [0, 0, 1, 0])                  # one step later, new symbols


def test_message_not_visible_same_step():
    env = TrafficJunctionEnv("medium", bits=1, add_rate=0.0)
    env.reset(seed=0)
    place(env, 0, 0); place(env, 1, 3)
    o0 = env._obs(1).copy()
    assert o0[env.inbox_slice()].sum() == 0                           # nothing sent yet


def test_comm_radius_limits_hearing():
    env = TrafficJunctionEnv("medium", bits=1, add_rate=0.0, comm_radius=2)
    env.reset(seed=0)
    place(env, 0, 0); place(env, 1, 6)                                # opposite entries, far apart
    o, *_ = env.step(acts(env, BRAKE, msgs=[1, 1] + [0] * 8))
    assert o["car_0"][env.inbox_slice()].sum() == 0


def test_zero_bits_has_no_inbox():
    assert TrafficJunctionEnv("medium", bits=0).n_sym == 0
    assert TrafficJunctionEnv("medium", bits=1)._obs_dim - TrafficJunctionEnv("medium", bits=0)._obs_dim == 2


def test_route_is_private():
    a, b = TrafficJunctionEnv("medium", add_rate=0.0), TrafficJunctionEnv("medium", add_rate=0.0)
    for env, r1 in ((a, 3), (b, 4)):
        env.reset(seed=0)
        place(env, 0, 0); place(env, 1, r1)                           # car 1: same entry, other route
    assert np.array_equal(a._obs(0), b._obs(0))
    assert not np.array_equal(a._obs(1), b._obs(1))


def test_alpha_mixing():
    env = TrafficJunctionEnv("medium", add_rate=0.0, alpha=1.0)
    env.reset(seed=0)
    place(env, 0, 0); place(env, 1, 3); env.tau[1] = 4
    _, r, *_ = env.step(acts(env, BRAKE))
    assert r["car_0"] == pytest.approx(-0.03) and r["car_1"] == pytest.approx(-0.03)   # team mean of -0.01, -0.05
    env = TrafficJunctionEnv("medium", add_rate=0.0, alpha=0.5)
    env.reset(seed=0)
    place(env, 0, 0); place(env, 1, 3); env.tau[1] = 4
    _, r, *_ = env.step(acts(env, BRAKE))
    assert r["car_0"] == pytest.approx(0.5 * -0.01 + 0.5 * -0.03)


def test_reset_determinism():
    out = []
    for _ in range(2):
        env = TrafficJunctionEnv("medium", add_rate=0.2)
        env.reset(seed=3)
        for _ in range(15):
            env.step(acts(env, GAS))
        out.append((env.stats.copy(), env.active.copy().tolist()))
    assert out[0] == out[1]


def test_recorder_and_viewer(tmp_path):
    import json
    from src.viz.record import ScriptedDriver, env_meta, record_episode
    from src.viz.replay import write_html
    env = TrafficJunctionEnv("medium", bits=1)
    ep = record_episode(env, ScriptedDriver("always_gas"), seed=5, add_rate=0.2)
    assert len(ep["frames"]) == env.max_steps + 1
    assert ep["summary"]["spawned"] == ep["frames"][-1]["total"]["spawned"]
    out = write_html(env_meta(env), [ep], str(tmp_path / "r.html"))
    html = open(out).read()
    assert "/*__DATA__*/" not in html
    blob = html.split('<script id="replay-data" type="application/json">')[1].split("</script>")[0]
    assert json.loads(blob.replace("<\\/", "</"))["meta"]["H"] == 14


def test_eval_interventions_change_only_inbox():
    from src.evaluate_tj import make_transform
    env = TrafficJunctionEnv("medium", bits=2, add_rate=0.0)
    env.reset(seed=0)
    place(env, 0, 0); place(env, 1, 3)
    env.step(acts(env, BRAKE, msgs=[2, 2] + [0] * 8))
    clean = env._obs(1).copy()
    env.inbox_transform = make_transform("permute", env.n_sym)
    shifted = env._obs(1)
    sl = env.inbox_slice()
    assert np.array_equal(clean[:sl.start], shifted[:sl.start])         # own features untouched
    assert clean[sl].argmax() == 2 and shifted[sl].argmax() == 3
