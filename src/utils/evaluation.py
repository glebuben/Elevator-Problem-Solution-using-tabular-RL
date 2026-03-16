"""Evaluation and result printing utilities."""

import numpy as np

from ..env import ElevatorEnv


def evaluate_policy(policy_fn, num_steps=10000, lam=0.05,
                    traffic='uniform', seed=42):
    env = ElevatorEnv(lam=lam, traffic=traffic, seed=seed)
    state = env.reset()
    total_reward = 0.0

    for _ in range(num_steps):
        va = env.get_valid_actions()
        action = policy_fn(state, va)
        state, reward = env.step(action)
        total_reward += reward

    cw = env.total_wait_time
    cb = env.total_boarded
    cs = env.total_system_time
    cv = env.total_served
    pw, pwn, ps, psn = env.get_pending_metrics()

    return {
        'avg_reward': total_reward / num_steps,
        'avg_wait_c': (cw / cb if cb > 0
                       else float('nan')),
        'avg_sys_c': (cs / cv if cv > 0
                      else float('nan')),
        'avg_wait_i': ((cw + pw) / (cb + pwn)
                       if (cb + pwn) > 0
                       else float('nan')),
        'avg_sys_i': ((cs + ps + pw) / (cv + psn + pwn)
                      if (cv + psn + pwn) > 0
                      else float('nan')),
        'served': cv,
        'boarded': cb,
        'still_waiting': pwn,
        'in_car': psn,
        'total_reward': total_reward,
    }


def _fmt(v, w=10, d=2):
    if isinstance(v, float) and np.isnan(v):
        return "N/A".rjust(w)
    if isinstance(v, float):
        return f"{v:.{d}f}".rjust(w)
    return str(v).rjust(w)


def print_results(results):
    print()
    hdr = (f"{'Policy':<16s} {'Rew/step':>9s} "
           f"{'Wait(c)':>9s} {'Sys(c)':>9s} "
           f"{'Wait(i)':>9s} {'Sys(i)':>9s} "
           f"{'Served':>7s} {'Board':>7s} "
           f"{'Queue':>7s} {'InCar':>7s}")
    print(hdr)
    print('-' * len(hdr))
    for name, m in results.items():
        print(f"{name:<16s} "
              f"{_fmt(m['avg_reward'],9,3)} "
              f"{_fmt(m['avg_wait_c'],9)} "
              f"{_fmt(m['avg_sys_c'],9)} "
              f"{_fmt(m['avg_wait_i'],9)} "
              f"{_fmt(m['avg_sys_i'],9)} "
              f"{_fmt(m['served'],7)} "
              f"{_fmt(m['boarded'],7)} "
              f"{_fmt(m['still_waiting'],7)} "
              f"{_fmt(m['in_car'],7)}")
    print("\n  (c)=completed  (i)=inclusive (+pending)")