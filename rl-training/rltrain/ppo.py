# -*- coding: utf-8 -*-
"""PPO(参考实现,torch)——Actor-Critic + GAE + 裁剪代理目标。

导出形态 = MLP(单输入单输出,契约 onnx_io)——循环网络见 algos/recurrent.yaml
的诚实边界(引擎 IO 单输入,hidden state 图不满足当前契约)。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class RunningNorm:
    """观测归一化——导出时烘焙进计算图(引擎不归一化)。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = eps
        self.eps = eps

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        m = x.mean(axis=0)
        v = x.var(axis=0)
        n = x.shape[0]
        delta = m - self.mean
        tot = self.count + n
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta ** 2 * self.count * n / tot) / tot
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + self.eps), -10, 10)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(256, 256)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        self.body = nn.Sequential(*layers)
        self.mu = nn.Linear(d, act_dim)
        self.log_std = nn.Parameter(torch.as_tensor(np.full(act_dim, -0.5, dtype=np.float32)))
        self.v = nn.Linear(d, 1)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.mu(h), self.v(h).squeeze(-1)

    def dist(self, obs: torch.Tensor) -> Normal:
        mu, _ = self.forward(obs)
        return Normal(mu, self.log_std.exp().expand_as(mu))


def gae(rew, done, val, last_val, gamma=0.99, lam=0.95) -> tuple[np.ndarray, np.ndarray]:
    """rew/done/val: [T, N];返回 (adv, ret) 同形。"""
    rew = np.asarray(rew, dtype=np.float32)
    done = np.asarray(done, dtype=np.float32)
    val = np.asarray(val, dtype=np.float32)
    t, n = rew.shape[:2]
    adv = np.zeros((t, n), dtype=np.float32)
    last = np.zeros(n, dtype=np.float32)
    for i in reversed(range(t)):
        nxt = last_val if i == t - 1 else val[i + 1]
        nonterm = 1.0 - done[i]
        delta = rew[i] + gamma * nxt * nonterm - val[i]
        last = adv[i] = delta + gamma * lam * nonterm * last
    return adv, adv + val


def updateppo(model: ActorCritic, opt: torch.optim.Optimizer, obs_norm, buf: dict,
              epochs: int = 4, minibatch: int = 512, clip: float = 0.2,
              vf_coef: float = 0.5, ent_coef: float = 0.0, device: str = "cpu") -> dict:
    """buf: obs[T,N,D] act[T,N,A] logp[T,N] adv[T,N] ret[T,N](numpy)。"""
    o = torch.as_tensor(buf["obs"], dtype=torch.float32, device=device)
    a = torch.as_tensor(buf["act"], dtype=torch.float32, device=device)
    adv = torch.as_tensor(buf["adv"], dtype=torch.float32, device=device)
    ret = torch.as_tensor(buf["ret"], dtype=torch.float32, device=device)
    logp_old = torch.as_tensor(buf["logp"], dtype=torch.float32, device=device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    t, n = o.shape[0], o.shape[1]
    flat = lambda x: x.reshape(t * n, *x.shape[2:])
    o, a, adv, ret, logp_old = map(flat, (o, a, adv, ret, logp_old))
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
    idx_all = np.arange(o.shape[0])
    for _ in range(epochs):
        np.random.shuffle(idx_all)
        for lo in range(0, len(idx_all), minibatch):
            idx = torch.as_tensor(idx_all[lo:lo + minibatch], device=device)
            dist = model.dist(o[idx])
            logp = dist.log_prob(a[idx]).sum(-1)
            ratio = (logp - logp_old[idx]).exp()
            pi_loss = -torch.min(ratio * adv[idx],
                                 torch.clamp(ratio, 1 - clip, 1 + clip) * adv[idx]).mean()
            _, v = model.forward(o[idx])
            v_loss = ((v - ret[idx]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats["pi_loss"] += float(pi_loss)
            stats["v_loss"] += float(v_loss)
            stats["entropy"] += float(ent)
    k = max(1, epochs * (o.shape[0] // minibatch))
    return {kk: vv / k for kk, vv in stats.items()}


@torch.no_grad()
def act(model: ActorCritic, obs: np.ndarray, obs_norm: RunningNorm,
        deterministic: bool = False, device: str = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    o = torch.as_tensor(obs_norm.normalize(obs), dtype=torch.float32, device=device)
    dist = model.dist(o)
    a_pre = dist.mean if deterministic else dist.sample()
    logp = dist.log_prob(a_pre).sum(-1)
    _, v = model.forward(o)
    a_env = torch.tanh(a_pre)   # [−1,1) → 环境端 neutral + a·scale
    # tanh 修正 log-prob
    logp = logp - torch.log(1 - a_env ** 2 + 1e-6).sum(-1)
    return a_env.cpu().numpy(), logp.cpu().numpy(), v.cpu().numpy()
