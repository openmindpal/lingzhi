#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练入口——策略族(PPO)与快脑(参考蒸馏)统一入口。

用法:
  python scripts/train.py --task family_balance --envs 16 --iters 2     (冒烟)
  python scripts/train.py --task family_reach  --algo ppo               (参考后端)
  python scripts/train.py --task family_reach --backend isaaclab        (GPU 大规模)

后端: mujoco(默认,单机即可跑) | isaaclab(需 Isaac Lab GPU 环境——诚实报错)。
产物: runs/<id>/{ckpt.pt, manifest.json, metrics.jsonl}
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # rl-training

import numpy as np
import torch

from rltrain.g1_cfg import G1Cfg
from rltrain.tasks import load_task
from rltrain.env_mj import G1VecEnv
from rltrain import ppo as P


def build_manifest(task: dict, cfg: G1Cfg, obs_dim: int, args, env=None) -> dict:
    mode = task["obs_mode"]
    if task.get("kind") == "fastbrain":
        return {"kind": "fastbrain", "task": task["name"], "name": task["row_name"],
                "weights_rel": "fastbrains/tactician.onnx", "goal_stream": task["goal_stream"],
                "rate_hz": float(task.get("rate_hz", 50.0)),
                "budget_ms": int(task.get("budget_ms", 60000)),
                "input_dim": obs_dim, "output_dim": 4}
    if env is not None:
        jn = list(env.joint_names)
        soft = {n: [float(env.soft_lo[i]), float(env.soft_hi[i])] for i, n in enumerate(jn)}
        neutral = {n: float(env.neutral[i]) for i, n in enumerate(jn)}
        ascale = [float(x) for x in env.action_scale_vec]
        j = int(env.j)
    else:
        jn = list(cfg.joints)
        soft = {n: list(cfg.limits[n]) for n in jn}
        neutral = {n: cfg.neutral.get(n, 0.0) for n in jn}
        ascale = [float(task.get("action_scale", 0.3))] * cfg.j
        j = cfg.j
    return {"kind": "policy", "task": task["name"], "family": task["family"],
            "row_name": task["row_name"], "weights_rel": f"policies/{task['row_name']}.onnx",
            "obs_mode": mode, "obs_dim": int(obs_dim), "j": j, "b": 3,
            "joint_names": jn, "soft_limits": soft, "neutral": neutral,
            "action_scale": ascale,
            "frequency_hz": float(task["control_hz"]), "priority": task.get("priority", "task"),
            "goal_conditioned": "goal" in mode, "obs_contact": "contact" in mode,
            "hidden": args.hidden, "seed": args.seed, "iters": args.iters,
            "backend": args.backend, "algo": args.algo, "device": args.device,
            "exports": []}


def train_policy(task: dict, cfg: G1Cfg, args, run_dir: str, env=None) -> None:
    from rltrain.isaaclab_env import IsaacLabVecEnv
    if env is None:
        env = G1VecEnv(cfg, task, args.envs, seed=args.seed)
    obs_dim = env.assembler.dim
    manifest = build_manifest(task, cfg, obs_dim, args, env=env)
    torch.manual_seed(args.seed)
    model = P.ActorCritic(obs_dim, env.j, hidden=tuple(args.hidden)).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    norm = P.RunningNorm(obs_dim)
    print(f"[train] family={task['family']} obs_dim={obs_dim} J={env.j} envs={args.envs} "
          f"backend={args.backend} device={args.device}")

    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(os.path.join(run_dir, "tb"))
    except Exception:
        pass   # tensorboard 可选——缺包不影响训练
    best = {"eval": -9e9, "iter": 0}
    eval_every = int(getattr(args, "eval_every", 10) or 0)

    # ── BC 预训练(il_pretrain 适配):演示 (obs,act) 对 → 策略热启动 ──
    # 演示 act 为 tanh 域(环境端 neutral+a·scale),BC 目标 = tanh(mu(norm(obs)))。
    # 预训练后接 RL 微调——产物走同一导出/打包/验收流水线(契约不变)。
    if getattr(args, "pretrain_demos", None):
        from rltrain.demos import load as load_demos
        dm = load_demos(args.pretrain_demos)
        if dm["obs"].shape[1] != obs_dim:
            raise SystemExit(
                f"[pretrain] 演示 obs 维度 {dm['obs'].shape[1]} ≠ 环境观测 {obs_dim}"
                "(演示须录自同任务同包——维度即契约)")
        norm.update(dm["obs"])   # 演示 obs 纳入归一化统计(与 RL 阶段同分布)
        X = torch.as_tensor(dm["obs"], dtype=torch.float32)
        Y = torch.as_tensor(dm["act"], dtype=torch.float32)
        n_dm = X.shape[0]
        bc_first = None
        for ep in range(max(1, args.bc_pretrain_epochs)):
            perm = torch.randperm(n_dm)
            tot = 0.0
            for lo in range(0, n_dm, 512):
                idx = perm[lo:lo + 512]
                with torch.no_grad():
                    xn = torch.as_tensor(norm.normalize(X[idx].numpy()), dtype=torch.float32)
                pred = torch.tanh(model.forward(xn)[0])
                loss = ((pred - Y[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                tot += loss.item()
            bc_first = bc_first if bc_first is not None else tot
            print(f"[pretrain] epoch {ep + 1}/{args.bc_pretrain_epochs} bc_loss={tot:.5f}")
        manifest["pretrain"] = {"demos": args.pretrain_demos, "samples": int(n_dm),
                                "epochs": args.bc_pretrain_epochs,
                                "bc_loss_first": round(bc_first, 5),
                                "bc_loss_last": round(tot, 5)}
        print(f"[pretrain] BC 预训练完成: {n_dm} 样本, loss {bc_first:.5f} → {tot:.5f}")

    def eval_quick() -> float:
        """确定性快速评估(验收门子集的代理判据——正式验收仍走 eval_gate)。"""
        obs_e = env.reset()
        tot = 0.0
        steps = min(60, env.max_ep_steps)
        with torch.no_grad():
            for _ in range(steps):
                o = torch.as_tensor(norm.normalize(obs_e), dtype=torch.float32,
                                    device=args.device)
                mu, _ = model.forward(o)
                a = torch.tanh(mu).cpu().numpy()
                obs_e, r, _, _ = env.step(a)
                tot += float(np.mean(r))
        return tot / steps

    obs = env.reset()
    lines = []
    for it in range(1, args.iters + 1):
        if args.push_curriculum and task.get("push"):
            env.set_curriculum(0.2 + 0.8 * it / max(1, args.iters))   # 扰动课程: 20%→100%
        T = args.steps
        B = {"obs": np.zeros((T, args.envs, obs_dim), dtype=np.float32),
             "act": np.zeros((T, args.envs, env.j), dtype=np.float32),
             "logp": np.zeros((T, args.envs), dtype=np.float32),
             "val": np.zeros((T, args.envs), dtype=np.float32)}
        rews = np.zeros((T, args.envs), dtype=np.float32)
        dones = np.zeros((T, args.envs), dtype=np.float32)
        obs_n = norm.normalize(obs)
        with torch.no_grad():
            a, logp, v = P.act(model, obs, norm, device=args.device)
        B["obs"][0], B["act"][0], B["logp"][0], B["val"][0] = obs_n, a, logp, v
        norm.update(obs)
        ep_ret = np.zeros(args.envs)
        for t in range(T):
            obs, r, done, infos = env.step(a)
            ep_ret += r
            rews[t], dones[t] = r, done
            for e, inf in enumerate(infos):
                if done[e] and "final_obs" in inf:
                    obs[e] = inf["final_obs"]
            with torch.no_grad():
                a, logp, v = P.act(model, obs, norm, device=args.device)
            if t + 1 < T:
                B["obs"][t + 1], B["act"][t + 1] = norm.normalize(obs), a
                B["logp"][t + 1], B["val"][t + 1] = logp, v
        norm.update(obs)
        with torch.no_grad():
            _, last_v = model.forward(torch.as_tensor(
                norm.normalize(obs), dtype=torch.float32, device=args.device))
        adv, ret = P.gae(rews, dones, B["val"], last_v.cpu().numpy())
        stats = P.updateppo(model, opt, norm,
                            {"obs": B["obs"], "act": B["act"], "logp": B["logp"],
                             "adv": adv, "ret": ret}, device=args.device)
        line = {"iter": it, "ep_ret_mean": float(ep_ret.mean()),
                "ep_ret_max": float(ep_ret.max()), **stats, "ts": time.time()}
        lines.append(line)
        if tb:
            tb.add_scalar("ep_ret", line["ep_ret_mean"], it)
            tb.add_scalar("pi_loss", stats["pi_loss"], it)
            tb.add_scalar("v_loss", stats["v_loss"], it)
        ev_txt = ""
        if eval_every and (it % eval_every == 0 or it == args.iters):
            ev = eval_quick()
            if tb:
                tb.add_scalar("eval", ev, it)
            ev_txt = f" eval={ev:8.2f}"
            if ev > best["eval"]:
                best = {"eval": ev, "iter": it}
                torch.save({"model": model.state_dict(),
                            "obs_mean": norm.mean, "obs_var": norm.var, "obs_count": norm.count},
                           os.path.join(run_dir, "ckpt_best.pt"))
                manifest["best"] = best
        print(f"[it {it:4d}] ret={line['ep_ret_mean']:8.2f} max={line['ep_ret_max']:8.2f} "
              f"pi={stats['pi_loss']:+.4f} v={stats['v_loss']:.4f}{ev_txt}")
        if it % 10 == 0 or it == args.iters:
            torch.save({"model": model.state_dict(),
                        "obs_mean": norm.mean, "obs_var": norm.var, "obs_count": norm.count},
                       os.path.join(run_dir, "ckpt.pt"))
            manifest["iters_done"] = it
            with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=1)
    with open(os.path.join(run_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    torch.save({"model": model.state_dict(),
                "obs_mean": norm.mean, "obs_var": norm.var, "obs_count": norm.count},
               os.path.join(run_dir, "ckpt.pt"))
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    if tb:
        tb.close()
    print(f"[done] {run_dir}/ckpt.pt (best: iter={best['iter']} eval={best['eval']:.2f})")


def train_fastbrain(task: dict, cfg: G1Cfg, args, run_dir: str) -> None:
    """参考蒸馏:解析追靶专家(限速/限域)生成 (obs+goal)→goal' 对,行为克隆。

    诚实边界: 生产快脑 = RL 外环(架构 §4.6 预留)——本路径验证 IO/约束/导出链。
    """
    from rltrain.tasks import load_task
    from rltrain.contract import base_width
    base_task = load_task(task["base_task"])
    env = G1VecEnv(cfg, base_task, args.envs, seed=args.seed)
    # 契约(fastbrain_contract): 输入 = 肌肉同源观测的**基础块**(97,q/dq/tau/imu)+当前 goal(4)
    bw = base_width(cfg.j)
    in_dim = bw + 4
    torch.manual_seed(args.seed)
    net = torch.nn.Sequential(torch.nn.Linear(in_dim, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 4))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xs, ys = [], []
    obs = env.reset()
    for _ in range(args.steps * 4):
        for e in range(args.envs):
            g = env.ep_goal[e]
            if g is None:
                continue
            g_cur = env._goal_obs(e)
            g_next = env.expert_goal(e)
            xs.append(np.concatenate([obs[e][:bw], g_cur]).astype(np.float32))
            ys.append(g_next.astype(np.float32))
        env.step(np.zeros((args.envs, cfg.j)))   # 肌肉中性保持——蒸馏只学 goal 动力学
        if len(xs) >= args.bc_samples:
            break
    X = torch.as_tensor(np.stack(xs))
    Y = torch.as_tensor(np.stack(ys))
    rate = float(task.get("rate_penalty", 0.1))
    for ep in range(args.bc_epochs):
        perm = torch.randperm(X.shape[0])
        for lo in range(0, len(perm), 256):
            idx = perm[lo:lo + 256]
            pred = net(X[idx])
            loss = ((pred - Y[idx]) ** 2).mean() + rate * ((pred - X[idx][:, -4:]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    print(f"[fastbrain] samples={len(X)} final_loss={float(loss):.6f}")
    torch.save({"net": net.state_dict(), "meta": {"in_dim": in_dim}}, os.path.join(run_dir, "ckpt.pt"))
    manifest = build_manifest(task, cfg, in_dim, args)
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)


def train_sac(task: dict, cfg: G1Cfg, args, run_dir: str) -> None:
    """SAC(力控类备选——algos/sac.yaml)。产物与 PPO 同 manifest/ckpt 形态。"""
    from rltrain.sac import SAC, Replay
    env = G1VecEnv(cfg, task, 1, seed=args.seed)   # off-policy 单环境采样
    obs_dim = env.assembler.dim
    agent = SAC(obs_dim, env.j, device=args.device, lr=float(args.lr),
                hidden=tuple(args.hidden))
    agent.buf = Replay(obs_dim, env.j)
    norm = P.RunningNorm(obs_dim)
    manifest = build_manifest(task, cfg, obs_dim, args, env=env)
    obs = env.reset()[0]
    lines = []
    total = max(args.sac_warmup + 1, args.steps * args.iters)
    for t in range(1, total + 1):
        if t <= args.sac_warmup:
            a = env.rs.uniform(-1, 1, env.j).astype(np.float32)
        else:
            a = agent.act(obs[None], norm)[0]
        o2, r, done, infos = env.step(a[None])
        o2, r, done, infos = o2[0], float(r[0]), bool(done[0]), infos[0]
        norm.update(obs[None])
        agent.buf.add(obs, a, r, o2, float(done))
        if done:
            obs = infos.get("final_obs", env.reset()[0])
        else:
            obs = o2
        if t > args.sac_warmup and t % 2 == 0:
            batch = agent.buf.sample(args.batch, env.rs)
            stats = agent.update(batch, norm)
            if t % 500 == 0:
                lines.append({"step": t, **stats})
                print(f"[sac {t:6d}] q={stats['q_loss']:.4f} pi={stats['pi_loss']:+.4f} "
                      f"alpha={stats['alpha']:.3f}")
    with open(os.path.join(run_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln) + "\n")
    torch.save({"model": agent.actor.state_dict(),
                "obs_mean": norm.mean, "obs_var": norm.var, "obs_count": norm.count},
               os.path.join(run_dir, "ckpt.pt"))
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"[done] {run_dir}/ckpt.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--backend", default="mujoco", choices=["mujoco", "isaaclab"])
    ap.add_argument("--device", default="cpu", help="cuda:0(GPU 服务器) / cpu")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--steps", type=int, default=128, help="每次迭代每环境步数")
    ap.add_argument("--lr", default="3e-4")
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=10, help="每 N 迭代快速评估+最优保存(0=关)")
    ap.add_argument("--pretrain-demos", default=None,
                    help="演示数据 npz(demos.record 产物)——BC 预训练策略热启动(il_pretrain)")
    ap.add_argument("--bc-pretrain-epochs", type=int, default=10,
                    help="BC 预训练轮数(仅在 --pretrain-demos 时生效)")
    ap.add_argument("--push-curriculum", action="store_true",
                    help="扰动课程: 推力幅值随迭代线性放大到任务设定值")
    ap.add_argument("--sac-warmup", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--bc-samples", type=int, default=4096)
    ap.add_argument("--bc-epochs", type=int, default=20)
    ap.add_argument("--out", default=None, help="runs/<id>(缺省按任务+时间生成)")
    args = ap.parse_args()
    task = load_task(args.task)
    cfg = G1Cfg()
    run_dir = args.out or os.path.join(os.path.dirname(_HERE), "runs",
                                       f"{args.task}-{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"[run] {run_dir}")
    if task.get("kind") == "fastbrain":
        train_fastbrain(task, cfg, args, run_dir)
    elif args.algo == "sac":
        if args.backend != "mujoco":
            raise SystemExit("sac 当前接 mujoco 后端(isaaclab 适配点)")
        train_sac(task, cfg, args, run_dir)
    else:
        if args.backend == "isaaclab":
            from rltrain.isaaclab_env import IsaacLabVecEnv
            env = IsaacLabVecEnv(cfg, task, args.envs, seed=args.seed, device=args.device)
            train_policy(task, cfg, args, run_dir, env=env)
        else:
            train_policy(task, cfg, args, run_dir)


if __name__ == "__main__":
    main()
