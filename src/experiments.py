#!/usr/bin/env python3
"""Run all experiments and save models.

Usage:
    python -m src.experiments
    python -m src.experiments --train_steps 2000000
    python -m src.experiments --only_groups primary mixed
    python -m src.experiments --skip_groups high_traffic
"""

import argparse
import time

import numpy as np

from .config import TrainConfig
from .train import make_agent, train_agent
from .compare import run_comparison
from .utils.storage import run_dir


# ─────────────────────────────────────────────────────
# Experiment definitions
# ─────────────────────────────────────────────────────

def default_config(**overrides):
    """Base configuration matching the original script."""
    defaults = dict(
        lam=0.001,
        traffic='uniform',
        gamma=0.99,
        alpha=0.1,
        train_steps=5_000_000,
        eval_steps=10_000,
        eval_seed=42,
        epsilon_start=1.0,
        epsilon_end=0.01,
        agent_seed=456,
        env_seed=123,
        reset_interval=50_000,
        eval_interval=200_000,
        checkpoint_interval=500_000,
        save_checkpoints=True,
        output_dir='results',
    )
    defaults.update(overrides)
    defaults['epsilon_decay_steps'] = defaults['train_steps'] // 2
    return defaults


EXPERIMENTS = [
    # Group 1: Primary (uniform, λ=0.001)
    {
        'group': 'primary',
        'description': (
            'Q-learning vs SARSA on uniform traffic (λ=0.001)'),
        'runs': [
            {'agent_type': 'q_learning',
             'traffic': 'uniform', 'lam': 0.001},
            {'agent_type': 'sarsa',
             'traffic': 'uniform', 'lam': 0.001},
        ],
        'compare': {'lam': 0.001, 'traffic': 'uniform'},
    },
    # Group 2: up_peak
    {
        'group': 'up_peak',
        'description': (
            'Q-learning vs SARSA on up_peak traffic (λ=0.001)'),
        'runs': [
            {'agent_type': 'q_learning',
             'traffic': 'up_peak', 'lam': 0.001},
            {'agent_type': 'sarsa',
             'traffic': 'up_peak', 'lam': 0.001},
        ],
        'compare': {'lam': 0.001, 'traffic': 'up_peak'},
    },
    # Group 3: down_peak
    {
        'group': 'down_peak',
        'description': (
            'Q-learning vs SARSA on down_peak traffic (λ=0.001)'),
        'runs': [
            {'agent_type': 'q_learning',
             'traffic': 'down_peak', 'lam': 0.001},
            {'agent_type': 'sarsa',
             'traffic': 'down_peak', 'lam': 0.001},
        ],
        'compare': {'lam': 0.001, 'traffic': 'down_peak'},
    },
    # Group 4: mixed
    {
        'group': 'mixed',
        'description': (
            'Q-learning vs SARSA on mixed traffic (λ=0.001)'),
        'runs': [
            {'agent_type': 'q_learning',
             'traffic': 'mixed', 'lam': 0.001},
            {'agent_type': 'sarsa',
             'traffic': 'mixed', 'lam': 0.001},
        ],
        'compare': {'lam': 0.001, 'traffic': 'mixed'},
    },
    # Group 5: High traffic
    {
        'group': 'high_traffic',
        'description': (
            'High traffic: Q-learning λ=0.01, '
            'SARSA λ=0.01 (uniform)'),
        'runs': [
            {'agent_type': 'q_learning',
             'traffic': 'uniform', 'lam': 0.01,
             'run_name': 'Q-learning_uniform_lam0.1'},
            {'agent_type': 'sarsa',
             'traffic': 'uniform', 'lam': 0.01,
             'run_name': 'SARSA_uniform_lam0.01'},
        ],
        'compare': {'lam': 0.01, 'traffic': 'uniform'},
    },
]


# ─────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────

def run_experiment_group(experiment, global_overrides):
    """Train all runs in a group, then compare."""
    group = experiment['group']
    desc = experiment['description']

    print(f"\n{'#' * 112}")
    print(f"  EXPERIMENT GROUP: {group}")
    print(f"  {desc}")
    print(f"{'#' * 112}")

    trained_paths = []

    for run_spec in experiment['runs']:
        cfg_dict = default_config(**global_overrides)
        cfg_dict.update(run_spec)

        if 'run_name' not in run_spec:
            agent_label = (
                'Q-learning'
                if cfg_dict['agent_type'] == 'q_learning'
                else 'SARSA')
            cfg_dict['run_name'] = (
                f"{agent_label}_{cfg_dict['traffic']}"
                f"_lam{cfg_dict['lam']}")

        cfg_dict['epsilon_decay_steps'] = (
            cfg_dict['train_steps'] // 2)

        cfg = TrainConfig.from_dict(cfg_dict)
        agent = make_agent(cfg)

        print(f"\n{'─' * 80}")
        print(f"  Run: {cfg.run_name}")
        print(f"  agent={cfg.agent_type}, λ={cfg.lam}, "
              f"traffic={cfg.traffic}, "
              f"steps={cfg.train_steps}")
        print(f"{'─' * 80}")

        train_agent(agent, cfg)

        best_path = (
            f"{run_dir(cfg.output_dir, cfg.run_name)}/best.npz")
        trained_paths.append(best_path)

    cmp = experiment.get('compare', {})
    if cmp:
        cmp_lam = cmp.get('lam', 0.001)
        cmp_traffic = cmp.get('traffic', 'uniform')
        cmp_eval_steps = global_overrides.get(
            'eval_steps', 10_000)
        cmp_eval_seed = global_overrides.get('eval_seed', 42)

        run_comparison(
            agent_paths=trained_paths,
            lam=cmp_lam,
            traffic=cmp_traffic,
            eval_steps=cmp_eval_steps,
            eval_seed=cmp_eval_seed,
        )

    return trained_paths


def run_all_experiments(global_overrides, skip_groups=None):
    """Run every experiment group sequentially."""
    skip_groups = set(skip_groups or [])
    all_paths = []
    total_t0 = time.time()

    print("=" * 112)
    print("ELEVATOR RL — FULL EXPERIMENT SUITE")
    print("=" * 112)

    scheduled = [e for e in EXPERIMENTS
                 if e['group'] not in skip_groups]
    skipped = [e for e in EXPERIMENTS
               if e['group'] in skip_groups]

    print(f"\nScheduled: {len(scheduled)} groups")
    for e in scheduled:
        print(f"  • {e['group']}: {e['description']}")
    if skipped:
        print(f"Skipped:   {len(skipped)} groups")
        for e in skipped:
            print(f"  ✗ {e['group']}")
    print()

    for i, experiment in enumerate(scheduled, 1):
        group_t0 = time.time()
        print(f"\n{'█' * 112}")
        print(f"  [{i}/{len(scheduled)}] Starting group: "
              f"{experiment['group']}")
        print(f"{'█' * 112}")

        paths = run_experiment_group(
            experiment, global_overrides)
        all_paths.extend(paths)

        group_time = time.time() - group_t0
        print(f"\n  Group '{experiment['group']}' completed "
              f"in {group_time:.1f}s")

    total_time = time.time() - total_t0
    print(f"\n{'=' * 112}")
    print(f"ALL EXPERIMENTS COMPLETED in {total_time:.1f}s "
          f"({total_time / 60:.1f} min)")
    print(f"{'=' * 112}")
    print(f"\nSaved models ({len(all_paths)}):")
    for p in all_paths:
        print(f"  {p}")
    print(f"\nTo compare any subset:")
    print(f"  python -m src.compare "
          f"{' '.join(all_paths[:2])} "
          f"--lam 0.001 --traffic uniform")

    return all_paths


# ─────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Run all elevator RL experiments')
    p.add_argument('--train_steps', type=int,
                   default=5_000_000,
                   help='Training steps per agent')
    p.add_argument('--eval_steps', type=int, default=10_000)
    p.add_argument('--eval_seed', type=int, default=42)
    p.add_argument('--output_dir', default='results')
    p.add_argument('--checkpoint_interval', type=int,
                   default=500_000)
    p.add_argument('--eval_interval', type=int, default=200_000)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--no_checkpoints', action='store_true',
                   help='Disable intermediate checkpoints')
    p.add_argument(
        '--skip_groups', nargs='*', default=[],
        choices=['primary', 'up_peak', 'down_peak',
                 'mixed', 'high_traffic'],
        help='Groups to skip')
    p.add_argument(
        '--only_groups', nargs='*', default=None,
        choices=['primary', 'up_peak', 'down_peak',
                 'mixed', 'high_traffic'],
        help='Run only these groups')
    return p.parse_args()


def main():
    args = parse_args()

    global_overrides = {
        'train_steps': args.train_steps,
        'eval_steps': args.eval_steps,
        'eval_seed': args.eval_seed,
        'output_dir': args.output_dir,
        'checkpoint_interval': args.checkpoint_interval,
        'eval_interval': args.eval_interval,
        'alpha': args.alpha,
        'gamma': args.gamma,
        'save_checkpoints': not args.no_checkpoints,
    }

    skip = set(args.skip_groups)
    if args.only_groups is not None:
        all_groups = {e['group'] for e in EXPERIMENTS}
        skip = all_groups - set(args.only_groups)

    run_all_experiments(global_overrides, skip_groups=skip)


if __name__ == '__main__':
    main()