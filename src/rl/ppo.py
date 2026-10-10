"""Independent PPO: one actor, one critic and one optimizer per car.

Each car's loss uses only its own reward r_i and its own GAE. No parameter
sharing, no pooled gradients, no centralized critic. Structure follows
CleanRL's ppo.py (clipped objective, GAE, advantage normalization, value clip off).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer(i, o, std=np.sqrt(2)):
    l = nn.Linear(i, o)
    nn.init.orthogonal_(l.weight, std)
    nn.init.constant_(l.bias, 0.0)
    return l


class CarPolicy(nn.Module):
    """Factorized policy: pi(move|o) * pi(msg|o), plus V_i(o) for this car's own return."""

    def __init__(self, obs_dim, n_sym, hidden=64):
        super().__init__()
        self.n_sym = n_sym
        self.body = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh())
        self.move_head = layer(hidden, 2, 0.01)
        self.msg_head = layer(hidden, n_sym, 0.01) if n_sym else None
        self.critic = nn.Sequential(layer(obs_dim, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh(),
                                    layer(hidden, 1, 1.0))

    def value(self, x):
        return self.critic(x).squeeze(-1)

    def dists(self, x):
        h = self.body(x)
        dm = Categorical(logits=self.move_head(h))
        dg = Categorical(logits=self.msg_head(h)) if self.msg_head is not None else None
        return dm, dg

    def act(self, x, move=None, msg=None, greedy=False):
        dm, dg = self.dists(x)
        if move is None:
            move = dm.probs.argmax(-1) if greedy else dm.sample()
        logp, ent_m = dm.log_prob(move), dm.entropy()
        ent_g = torch.zeros_like(ent_m)
        if dg is not None:
            if msg is None:
                msg = dg.probs.argmax(-1) if greedy else dg.sample()
            logp = logp + dg.log_prob(msg)
            ent_g = dg.entropy()
        else:
            msg = torch.zeros_like(move)
        return move, msg, logp, ent_m, ent_g


@dataclass
class PPOConfig:
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_move: float = 0.01
    ent_msg: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatches: int = 4


def gae(rew, val, done, valid, last_val, last_alive, gamma, lam):
    """Arrays [T, E]. `valid`=car was active at that step. `done`=car's episode ended at that step
    (arrival, crash or horizon). Invalid steps are skipped. Returns advantages and returns."""
    T, E = rew.shape
    adv = np.zeros((T, E), np.float32)
    last = np.zeros(E, np.float32)
    nxt_val = last_val * last_alive
    for t in reversed(range(T)):
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nxt_val * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        last = np.where(valid[t], last, 0.0)
        adv[t] = last
        nxt_val = np.where(valid[t], val[t], 0.0)
    return adv, adv + val


def ppo_update(policy, opt, batch, cfg: PPOConfig):
    obs, move, msg, logp_old, adv, ret = (batch[k] for k in ("obs", "move", "msg", "logp", "adv", "ret"))
    n = obs.shape[0]
    idx = np.arange(n)
    stats = []
    for _ in range(cfg.epochs):
        np.random.shuffle(idx)
        for j in np.array_split(idx, cfg.minibatches):
            if len(j) < 2:
                continue
            _, _, logp, ent_m, ent_g = policy.act(obs[j], move[j], msg[j])
            ratio = (logp - logp_old[j]).exp()
            a = adv[j]
            a = (a - a.mean()) / (a.std() + 1e-8)
            pg = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
            v = policy.value(obs[j])
            vl = 0.5 * ((v - ret[j]) ** 2).mean()
            loss = pg + cfg.vf_coef * vl - cfg.ent_move * ent_m.mean() - cfg.ent_msg * ent_g.mean()
            opt.zero_grad()
            loss.backward()
            msg_grad = 0.0
            if policy.msg_head is not None:
                msg_grad = float(policy.msg_head.weight.grad.norm())
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            opt.step()
            stats.append((pg.item(), vl.item(), ent_m.mean().item(), ent_g.mean().item(), msg_grad,
                          ((ratio - 1).abs() > cfg.clip).float().mean().item()))
    s = np.array(stats).mean(0)
    return dict(pg_loss=s[0], v_loss=s[1], ent_move=s[2], ent_msg=s[3], msg_grad_norm=s[4], clipfrac=s[5])
