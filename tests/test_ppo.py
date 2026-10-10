import numpy as np
import torch

from src.rl.ppo import CarPolicy, gae


def naive_gae(rew, val, done, valid, last_val, last_alive, g, lam):
    T, E = rew.shape
    adv = np.zeros((T, E))
    for e in range(E):
        idx = [t for t in range(T) if valid[t, e]]
        for k, t in enumerate(idx):
            a, disc = 0.0, 1.0
            for u in idx[k:]:
                nxt = 0.0
                if not done[u, e]:
                    nxt = val[u + 1, e] if u + 1 < T else last_val[e] * last_alive[e]
                a += disc * (rew[u, e] + g * nxt - val[u, e])
                if done[u, e]:
                    break
                disc *= g * lam
            adv[t, e] = a
    return adv


def test_gae_matches_naive_with_masks():
    rng = np.random.default_rng(0)
    T, E = 12, 3
    rew = rng.normal(size=(T, E)).astype(np.float32)
    val = rng.normal(size=(T, E)).astype(np.float32)
    done = np.zeros((T, E), np.float32)
    valid = np.ones((T, E), bool)
    done[4, 0] = 1; valid[5:7, 0] = False            # car done, inactive, then new episode
    done[8, 1] = 1
    valid[:, 2] = True
    last_val = rng.normal(size=E).astype(np.float32)
    last_alive = np.array([1, 1, 0], np.float32)
    done[T - 1, 2] = 1
    adv, _ = gae(rew, val, done, valid, last_val, last_alive, 0.99, 0.95)
    ref = naive_gae(rew, val, done, valid, last_val, last_alive, 0.99, 0.95)
    assert np.allclose(adv, ref, atol=1e-5)


def test_factorized_logprob():
    p = CarPolicy(10, 4)
    x = torch.randn(5, 10)
    m, g, lp, *_ = p.act(x)
    dm, dg = p.dists(x)
    assert torch.allclose(lp, dm.log_prob(m) + dg.log_prob(g))


def test_separate_parameters():
    a, b = CarPolicy(10, 2), CarPolicy(10, 2)
    assert all(pa.data_ptr() != pb.data_ptr() for pa, pb in zip(a.parameters(), b.parameters()))
