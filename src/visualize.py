#!/usr/bin/env python3
"""Visualize training progress from checkpoints and eval history.

Usage:
    python -m src.visualize results/checkpoints/Q-learning_uniform_lam0.05
    python -m src.visualize results/checkpoints/Q-learning_uniform_lam0.05 \
                            --checkpoint_eval --lam 0.05
"""

import argparse
import json
import os

import numpy as np

from .agents import QLearningAgent, SARSAAgent
from .utils.evaluation import evaluate_policy
from .utils.storage import load_checkpoint, list_checkpoints

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not found. Text output only.")


def load_eval_history(run_path):
    hist_path = os.path.join(run_path, 'eval_history.json')
    if not os.path.exists(hist_path):
        return None
    with open(hist_path, 'r') as f:
        return json.load(f)


def plot_training_curves(run_paths, save_path=None):
    if not HAS_MPL:
        print("Cannot plot without matplotlib.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for rp in run_paths:
        history = load_eval_history(rp)
        if history is None:
            print(f"  No eval_history.json in {rp}, skipping")
            continue

        label = os.path.basename(rp)
        steps = [e['step'] for e in history]
        rewards = [e['avg_reward'] for e in history]
        waits = [e['avg_wait_i'] for e in history]

        axes[0].plot(steps, rewards, label=label, marker='.')
        axes[1].plot(steps, waits, label=label, marker='.')

    axes[0].set_xlabel('Training Step')
    axes[0].set_ylabel('Avg Reward / Step')
    axes[0].set_title('Training Progress: Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Avg Wait Time (inclusive)')
    axes[1].set_title('Training Progress: Wait Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()


def evaluate_checkpoints(run_path, lam, traffic,
                         eval_steps=10000, eval_seed=42):
    ckpts = list_checkpoints(
        os.path.dirname(run_path),
        os.path.basename(run_path))

    if not ckpts:
        parts = run_path.rstrip('/').rsplit('/', 1)
        if len(parts) == 2:
            ckpts = list_checkpoints(parts[0], parts[1])

    if not ckpts:
        print(f"No checkpoints found for {run_path}")
        return []

    print(f"\nEvaluating {len(ckpts)} checkpoints "
          f"from {run_path}")
    print(f"  λ={lam}, traffic={traffic}, "
          f"eval_steps={eval_steps}")

    results = []
    for cp in ckpts:
        q_table, metadata = load_checkpoint(cp)
        agent_type = metadata.get('agent_type', 'Q-learning')

        if agent_type == 'SARSA':
            agent = SARSAAgent()
        else:
            agent = QLearningAgent()
        agent.load_q_table(q_table)
        agent.set_rng(np.random.RandomState(eval_seed + 10))

        m = evaluate_policy(
            lambda s, va, a=agent: a.select_action(
                s, va, greedy=True),
            eval_steps, lam, traffic, eval_seed)

        step = metadata.get('step', 0)
        print(f"  Step {step:>8d}: "
              f"rew/s={m['avg_reward']:.3f}, "
              f"wait_i={m['avg_wait_i']:.2f}, "
              f"served={m['served']}")

        results.append({'step': step, **m})

    return results


def plot_q_value_heatmap(checkpoint_path, save_path=None):
    if not HAS_MPL:
        print("Cannot plot without matplotlib.")
        return

    q_table, metadata = load_checkpoint(checkpoint_path)
    step = metadata.get('step', '?')
    agent_type = metadata.get('agent_type', '?')

    action_names = ['MOVE_UP', 'MOVE_DOWN', 'STOP', 'WAIT']
    q_init = metadata.get('q_init', -5.0)
    visited_mask = np.any(
        np.abs(q_table - q_init) > 0.01, axis=1)
    n_visited = visited_mask.sum()

    if n_visited == 0:
        print("No states visited yet.")
        return

    visited_q = q_table[visited_mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    max_q = visited_q.max(axis=1)
    axes[0].hist(max_q, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Max Q-value')
    axes[0].set_ylabel('Count')
    axes[0].set_title(
        f'{agent_type} step {step}: '
        f'Max Q distribution ({n_visited} states)')

    mean_q = visited_q.mean(axis=0)
    bars = axes[1].bar(
        action_names, mean_q, edgecolor='black')
    axes[1].set_ylabel('Mean Q-value')
    axes[1].set_title(
        f'{agent_type} step {step}: Mean Q by Action')
    for bar, val in zip(bars, mean_q):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{val:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        plt.show()


def parse_args():
    p = argparse.ArgumentParser(
        description='Visualize elevator RL training')
    p.add_argument('run_paths', nargs='+',
                   help='Run directories to visualize')
    p.add_argument('--checkpoint_eval', action='store_true',
                   help='Re-evaluate all checkpoints')
    p.add_argument('--q_heatmap',
                   help='Path to .npz for Q-value heatmap')
    p.add_argument('--lam', type=float, default=0.05)
    p.add_argument('--traffic', default='uniform')
    p.add_argument('--eval_steps', type=int, default=10_000)
    p.add_argument('--eval_seed', type=int, default=42)
    p.add_argument('--save', default=None,
                   help='Save plot path instead of showing')
    return p.parse_args()


def main():
    args = parse_args()

    if not args.checkpoint_eval and not args.q_heatmap:
        plot_training_curves(
            args.run_paths, save_path=args.save)

    if args.checkpoint_eval:
        for rp in args.run_paths:
            evaluate_checkpoints(
                rp, args.lam, args.traffic,
                args.eval_steps, args.eval_seed)

    if args.q_heatmap:
        plot_q_value_heatmap(
            args.q_heatmap, save_path=args.save)


if __name__ == '__main__':
    main()