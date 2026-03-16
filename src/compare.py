#!/usr/bin/env python3
"""Load trained agents and compare against baselines.

Usage:
    python -m src.compare results/Q-learning_uniform_lam0.001/best.npz
    python -m src.compare results/*/best.npz --lam 0.001
"""

import argparse
import os

import numpy as np

from .agents import QLearningAgent, SARSAAgent
from .policies import (
    RandomPolicy, SCANPolicy, LOOKPolicy, NearestCallPolicy,
)
from .utils.evaluation import evaluate_policy, print_results
from .utils.storage import load_checkpoint


def load_agent_from_file(path):
    q_table, metadata = load_checkpoint(path)
    agent_type = metadata.get('agent_type', 'Q-learning')
    if agent_type == 'SARSA':
        agent = SARSAAgent(
            alpha=metadata.get('alpha', 0.1),
            gamma=metadata.get('gamma', 0.99))
    else:
        agent = QLearningAgent(
            alpha=metadata.get('alpha', 0.1),
            gamma=metadata.get('gamma', 0.99))
    agent.load_q_table(q_table)
    agent.steps = metadata.get('steps_trained', 0)
    return agent, metadata


def run_comparison(agent_paths, lam=0.001, traffic='uniform',
                   eval_steps=10000, eval_seed=42):
    print(f"\n{'=' * 112}")
    print(f"COMPARISON: λ={lam}, traffic={traffic}, "
          f"steps={eval_steps}")
    print(f"{'=' * 112}")

    results = {}

    rng1 = np.random.RandomState(eval_seed + 1)
    rp = RandomPolicy(rng1)
    results['Random'] = evaluate_policy(
        rp.select_action, eval_steps, lam, traffic, eval_seed)

    sp = SCANPolicy()
    results['SCAN'] = evaluate_policy(
        sp.select_action, eval_steps, lam, traffic, eval_seed)

    lp = LOOKPolicy()
    results['LOOK'] = evaluate_policy(
        lp.select_action, eval_steps, lam, traffic, eval_seed)

    rng2 = np.random.RandomState(eval_seed + 2)
    nc = NearestCallPolicy(rng2)
    results['Nearest'] = evaluate_policy(
        nc.select_action, eval_steps, lam, traffic, eval_seed)

    for path in agent_paths:
        agent, metadata = load_agent_from_file(path)
        agent.set_rng(np.random.RandomState(eval_seed + 10))

        parent = os.path.basename(os.path.dirname(path))
        fname = os.path.basename(path).replace('.npz', '')
        label = f"{parent}/{fname}"
        if len(label) > 16:
            label = parent[:14] + '..'

        step = metadata.get('step', '?')
        train_lam = metadata.get('lam', '?')
        train_traffic = metadata.get('traffic', '?')
        print(f"\n  Loaded: {path}")
        print(f"    type={metadata.get('agent_type', '?')}, "
              f"step={step}, train_lam={train_lam}, "
              f"train_traffic={train_traffic}")

        results[label] = evaluate_policy(
            lambda s, va, a=agent: a.select_action(
                s, va, greedy=True),
            eval_steps, lam, traffic, eval_seed)

    print_results(results)
    return results


def parse_args():
    p = argparse.ArgumentParser(
        description='Compare trained agents against baselines')
    p.add_argument('paths', nargs='*',
                   help='Paths to .npz checkpoint files')
    p.add_argument('--lam', type=float, default=0.05)
    p.add_argument('--traffic', default='uniform',
                   choices=['uniform', 'up_peak',
                            'down_peak', 'mixed'])
    p.add_argument('--eval_steps', type=int, default=10_000)
    p.add_argument('--eval_seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    run_comparison(
        agent_paths=args.paths,
        lam=args.lam,
        traffic=args.traffic,
        eval_steps=args.eval_steps,
        eval_seed=args.eval_seed,
    )


if __name__ == '__main__':
    main()