"""Sanity gate: the same PPO code (CarPolicy + gae + ppo_update) must solve CartPole-v1."""
import sys

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, ".")
from src.rl.ppo import CarPolicy, PPOConfig, gae, ppo_update  # noqa: E402

torch.manual_seed(0)
E, T, U = 8, 128, 100
envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(E)],
                                autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
pol = CarPolicy(4, 0)
cfg = PPOConfig(lr=2.5e-4, ent_move=0.01)
opt = torch.optim.Adam(pol.parameters(), lr=cfg.lr, eps=1e-5)
obs, _ = envs.reset(seed=0)
ep_ret, done_rets = np.zeros(E), []
for u in range(U):
    O = np.zeros((T, E, 4), np.float32); A = np.zeros((T, E), np.int64); L = np.zeros((T, E), np.float32)
    V = np.zeros((T, E), np.float32); R = np.zeros((T, E), np.float32); D = np.zeros((T, E), np.float32)
    for t in range(T):
        x = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            m, _, lp, *_ = pol.act(x)
            v = pol.value(x)
        O[t], A[t], L[t], V[t] = obs, m.numpy(), lp.numpy(), v.numpy()
        obs, r, te, tr, info = envs.step(m.numpy())
        R[t], D[t] = r, (te | tr)
        ep_ret += r
        for e in np.where(te | tr)[0]:
            done_rets.append(ep_ret[e]); ep_ret[e] = 0
    with torch.no_grad():
        lv = pol.value(torch.as_tensor(obs, dtype=torch.float32)).numpy()
    adv, ret = gae(R, V, D, np.ones((T, E), bool), lv, np.ones(E, np.float32), cfg.gamma, cfg.gae_lambda)
    batch = dict(obs=torch.as_tensor(O.reshape(-1, 4)), move=torch.as_tensor(A.reshape(-1)),
                 msg=torch.zeros(T * E, dtype=torch.int64), logp=torch.as_tensor(L.reshape(-1)),
                 adv=torch.as_tensor(adv.reshape(-1)), ret=torch.as_tensor(ret.reshape(-1)))
    ppo_update(pol, opt, batch, cfg)
    if (u + 1) % 20 == 0:
        print(f"update {u+1}: mean return of last 20 episodes = {np.mean(done_rets[-20:]):.1f}", flush=True)
final = np.mean(done_rets[-20:])
print("PASS" if final >= 195 else "FAIL", final)
