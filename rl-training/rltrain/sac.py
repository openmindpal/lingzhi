# -*- coding: utf-8 -*-
"""SAC(参考实现,torch)——连续控制off-policy,力控类备选(algos/sac.yaml)。

导出对齐: actor 复用 ActorCritic 结构(mu 头)——导出 = tanh(mu) 绝对目标,
与 PPO 同一 export_onnx 路径(契约 onnx_io 不变)。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rltrain.ppo import ActorCritic, RunningNorm


class Replay:
    def __init__(self, obs_dim: int, act_dim: int, cap: int = 200000):
        self.o = np.zeros((cap, obs_dim), dtype=np.float32)
        self.a = np.zeros((cap, act_dim), dtype=np.float32)
        self.r = np.zeros(cap, dtype=np.float32)
        self.o2 = np.zeros((cap, obs_dim), dtype=np.float32)
        self.d = np.zeros(cap, dtype=np.float32)
        self.cap, self.ptr, self.n = cap, 0, 0

    def add(self, o, a, r, o2, d) -> None:
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.ptr = (self.ptr + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch: int, rs) -> tuple:
        idx = rs.randint(0, self.n, size=batch)
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx]


class SAC:
    def __init__(self, obs_dim: int, act_dim: int, device: str = "cpu",
                 lr: float = 3e-4, hidden=(256, 256), gamma: float = 0.99, tau: float = 0.005):
        self.device = device
        self.actor = ActorCritic(obs_dim, act_dim, hidden=hidden).to(device)
        self.q1 = self._q(obs_dim, act_dim, hidden).to(device)
        self.q2 = self._q(obs_dim, act_dim, hidden).to(device)
        self.q1_t = self._q(obs_dim, act_dim, hidden).to(device)
        self.q2_t = self._q(obs_dim, act_dim, hidden).to(device)
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.a_opt.add_param_group({"params": [self.log_alpha], "lr": lr})
        self.target_entropy = -float(act_dim)
        self.gamma, self.tau = gamma, tau

    class _Q(nn.Module):
        def __init__(self, o, a, hidden):
            super().__init__()
            layers, d = [], o + a
            for h in hidden:
                layers += [nn.Linear(d, h), nn.ReLU()]
                d = h
            layers.append(nn.Linear(d, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, o, a):
            return self.net(torch.cat([o, a], -1)).squeeze(-1)

    def _q(self, o, a, hidden):
        return self._Q(o, a, hidden)

    def act(self, obs: np.ndarray, norm: RunningNorm, deterministic=False) -> np.ndarray:
        with torch.no_grad():
            o = torch.as_tensor(norm.normalize(obs), dtype=torch.float32, device=self.device)
            mu, _ = self.actor.forward(o)
            if deterministic:
                return torch.tanh(mu).cpu().numpy()
            a = torch.tanh(mu + torch.exp(self.actor.log_std).expand_as(mu) *
                           torch.randn_like(mu))
            return a.cpu().numpy()

    def update(self, batch, norm: RunningNorm) -> dict:
        o, a, r, o2, d = batch
        O = torch.as_tensor(o, device=self.device)
        A = torch.as_tensor(a, device=self.device)
        R = torch.as_tensor(r, device=self.device)
        O2 = torch.as_tensor(o2, device=self.device)
        D = torch.as_tensor(d, device=self.device)
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            mu2, _ = self.actor.forward(O2)
            noise = torch.exp(self.actor.log_std).expand_as(mu2) * torch.randn_like(mu2)
            a2 = torch.tanh(mu2 + noise)
            q_t = torch.min(self.q1_t(O2, a2), self.q2_t(O2, a2))
            y = R + self.gamma * (1 - D) * (q_t - alpha * self._logp(O2, a2))
        q1, q2 = self.q1(O, A), self.q2(O, A)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()
        mu, _ = self.actor.forward(O)
        noise = torch.exp(self.actor.log_std).expand_as(mu) * torch.randn_like(mu)
        a_new = torch.tanh(mu + noise)
        pi_loss = (alpha * self._logp(O, a_new) -
                   torch.min(self.q1(O, a_new), self.q2(O, a_new))).mean()
        self.a_opt.zero_grad(); pi_loss.backward(); self.a_opt.step()
        alpha_loss = -(self.log_alpha.exp() *
                       (self._logp(O, a_new).detach() + self.target_entropy)).mean()
        self.a_opt.zero_grad(); alpha_loss.backward(); self.a_opt.step()
        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
            for p, pt in zip(self.q2.parameters(), self.q2_t.parameters()):
                pt.mul_(1 - self.tau).add_(self.tau * p)
        return {"q_loss": float(q_loss), "pi_loss": float(pi_loss),
                "alpha": float(self.log_alpha.exp())}

    def _logp(self, o, a) -> torch.Tensor:
        """tanh 高斯对数密度(数值稳定)。"""
        mu, _ = self.actor.forward(o)
        std = torch.exp(self.actor.log_std).expand_as(mu)
        atanh = torch.atanh(torch.clamp(a, -1 + 1e-6, 1 - 1e-6))
        logp = (-0.5 * ((atanh - mu) / std) ** 2 - std.log() -
                0.5 * np.log(2 * np.pi)).sum(-1)
        return (logp - torch.log(1 - a ** 2 + 1e-6).sum(-1)).sum(-1)
