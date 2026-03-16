#!/usr/bin/env python3
"""Epsilon decay ablation study on mixed traffic.

Compares three epsilon strategies:
  1. No decay:       ε stays constant throughout training
  2. Linear decay:   ε decays linearly (current default approach)
  3. Constant decay:  ε decays exponentially (multiplicative)

Each strategy is tested with both Q-learning and SARSA.

Usage:
    python -m src.experiments_epsilon
    python -m src.experiments_epsilon --train_steps 2000000
    python -m src.experiments_epsilon --epsilons 1.0 0.5 0.1 0.01
"""

import argparse
import json
import time
import os

import numpy as np

from .config import TrainConfig
from .env import ElevatorEnv
from .env.constants import NUM_ACTIONS
from .agents import QLearningAgent, SARSAAgent, TOTAL_STATES
from .agents.base_agent import BaseAgent
from .agents.state_encoding import state_to_index
from .policies import RandomPolicy, SCANPolicy, LOOKPolicy, NearestCallPolicy
from .utils.evaluation import evaluate_policy, print_results
from .utils.storage import (
    save_checkpoint, save_config, run_dir, _ensure_dir,
)


# ─────────────────────────────────────────────────────
# 1. CUSTOM EPSILON SCHEDULES
# ─────────────────────────────────────────────────────

class EpsilonSchedule:
    """Base class for epsilon schedules."""
    def __init__(self, name):
        self.schedule_name = name

    def get_epsilon(self, step, total_steps):
        raise NotImplementedError

    def to_dict(self):
        return {'schedule_name': self.schedule_name}


class NoDecaySchedule(EpsilonSchedule):
    """Constant epsilon throughout training."""
    def __init__(self, epsilon):
        super().__init__(f'no_decay_eps{epsilon}')
        self.epsilon = epsilon

    def get_epsilon(self, step, total_steps):
        return self.epsilon

    def to_dict(self):
        return {
            'schedule_name': self.schedule_name,
            'schedule_type': 'no_decay',
            'epsilon': self.epsilon,
        }


class LinearDecaySchedule(EpsilonSchedule):
    """Linear decay from eps_start to eps_end over decay_steps."""
    def __init__(self, eps_start=1.0, eps_end=0.01,
                 decay_fraction=0.5):
        super().__init__(
            f'linear_{eps_start}_to_{eps_end}_'
            f'frac{decay_fraction}')
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_fraction = decay_fraction

    def get_epsilon(self, step, total_steps):
        decay_steps = int(total_steps * self.decay_fraction)
        if decay_steps == 0:
            return self.eps_end
        frac = min(1.0, step / decay_steps)
        return (self.eps_start
                + (self.eps_end - self.eps_start) * frac)

    def to_dict(self):
        return {
            'schedule_name': self.schedule_name,
            'schedule_type': 'linear_decay',
            'eps_start': self.eps_start,
            'eps_end': self.eps_end,
            'decay_fraction': self.decay_fraction,
        }


class ExponentialDecaySchedule(EpsilonSchedule):
    """Exponential (multiplicative) decay: ε *= decay_rate every step.

    ε(t) = max(eps_end, eps_start * decay_rate^t)
    """
    def __init__(self, eps_start=1.0, eps_end=0.01,
                 decay_rate=0.999999):
        super().__init__(
            f'exp_{eps_start}_rate{decay_rate}')
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_rate = decay_rate

    def get_epsilon(self, step, total_steps):
        return max(
            self.eps_end,
            self.eps_start * (self.decay_rate ** step))

    @classmethod
    def from_target(cls, eps_start, eps_end, target_steps):
        """Compute decay_rate so that eps_end is reached at
        target_steps."""
        if target_steps <= 0 or eps_start <= 0 or eps_end <= 0:
            return cls(eps_start, eps_end, 0.999999)
        rate = (eps_end / eps_start) ** (1.0 / target_steps)
        return cls(eps_start, eps_end, rate)

    def to_dict(self):
        return {
            'schedule_name': self.schedule_name,
            'schedule_type': 'exponential_decay',
            'eps_start': self.eps_start,
            'eps_end': self.eps_end,
            'decay_rate': self.decay_rate,
        }


# ─────────────────────────────────────────────────────
# 2. AGENTS WITH CUSTOM SCHEDULES
# ─────────────────────────────────────────────────────

class ScheduledQLearning(QLearningAgent):
    """Q-learning with pluggable epsilon schedule."""
    def __init__(self, schedule, total_steps, **kwargs):
        super().__init__(**kwargs)
        self.schedule = schedule
        self.total_steps = total_steps

    @property
    def epsilon(self):
        return self.schedule.get_epsilon(
            self.steps, self.total_steps)

    def get_hyperparameters(self):
        hp = super().get_hyperparameters()
        hp['epsilon_schedule'] = self.schedule.to_dict()
        return hp


class ScheduledSARSA(SARSAAgent):
    """SARSA with pluggable epsilon schedule."""
    def __init__(self, schedule, total_steps, **kwargs):
        super().__init__(**kwargs)
        self.schedule = schedule
        self.total_steps = total_steps

    @property
    def epsilon(self):
        return self.schedule.get_epsilon(
            self.steps, self.total_steps)

    def get_hyperparameters(self):
        hp = super().get_hyperparameters()
        hp['epsilon_schedule'] = self.schedule.to_dict()
        return hp


# ─────────────────────────────────────────────────────
# 3. TRAINING LOOP (schedule-aware)
# ─────────────────────────────────────────────────────

def train_scheduled_agent(agent, cfg, run_name):
    """Train agent with custom epsilon schedule, saving
    checkpoints and eval history."""
    is_sarsa = isinstance(agent, SARSAAgent)

    save_config(cfg, cfg.output_dir, run_name)

    print(f"\n  Training {agent.name}: {cfg.train_steps} steps")
    print(f"    λ={cfg.lam}, traffic={cfg.traffic}")
    print(f"    Schedule: {agent.schedule.schedule_name}")
    print(f"    Output: {run_dir(cfg.output_dir, run_name)}")

    env = ElevatorEnv(
        lam=cfg.lam, traffic=cfg.traffic, seed=cfg.env_seed)
    state = env.reset()
    t0 = time.time()

    best_reward = -float('inf')
    best_step = 0

    va = env.get_valid_actions()
    action = agent.select_action(state, va)

    eval_history = []

    for step in range(cfg.train_steps):
        if step > 0 and step % cfg.reset_interval == 0:
            state = env.reset()
            va = env.get_valid_actions()
            action = agent.select_action(state, va)

        next_state, reward = env.step(action)
        next_valid = env.get_valid_actions()
        next_action = agent.select_action(
            next_state, next_valid)
        agent.update(
            state, action, reward, next_state,
            next_valid, next_action=next_action)
        state = next_state
        action = next_action

        current_step = step + 1

        # Checkpoint
        if (cfg.save_checkpoints
                and current_step % cfg.checkpoint_interval == 0):
            meta = {
                **agent.get_hyperparameters(),
                **cfg.to_dict(),
                'step': current_step,
                'epsilon_current': agent.epsilon,
                'run_name': run_name,
            }
            save_checkpoint(
                agent.Q, meta, cfg.output_dir, run_name,
                step=current_step)

        # Evaluation
        if current_step % cfg.eval_interval == 0:
            elapsed = time.time() - t0
            m = evaluate_policy(
                lambda s, va, a=agent: a.select_action(
                    s, va, greedy=True),
                cfg.eval_steps, cfg.lam, cfg.traffic,
                cfg.eval_seed)

            cr = m['avg_reward']
            is_best = cr > best_reward
            if is_best:
                best_reward = cr
                best_step = current_step
                meta = {
                    **agent.get_hyperparameters(),
                    **cfg.to_dict(),
                    'step': current_step,
                    'epsilon_current': agent.epsilon,
                    'eval_metrics': m,
                    'run_name': run_name,
                }
                save_checkpoint(
                    agent.Q, meta, cfg.output_dir, run_name,
                    is_best=True)

            eval_entry = {
                'step': current_step,
                'epsilon': agent.epsilon,
                'avg_reward': cr,
                'avg_wait_i': m['avg_wait_i'],
                'served': m['served'],
                'is_best': is_best,
            }
            eval_history.append(eval_entry)

            total_pax = (m['served'] + m['still_waiting']
                         + m['in_car'])
            mark = " *** BEST" if is_best else ""
            print(f"    Step {current_step:>8d} | "
                  f"ε={agent.epsilon:.4f} | "
                  f"rew/s={cr:.3f} | "
                  f"wait_i={m['avg_wait_i']:.2f} | "
                  f"served={m['served']:>4d}/"
                  f"{total_pax:>4d} | "
                  f"{elapsed:.1f}s{mark}")

    total_time = time.time() - t0
    print(f"  {agent.name} done in {total_time:.1f}s, "
          f"best step {best_step}, "
          f"reward={best_reward:.4f}")

    rd = run_dir(cfg.output_dir, run_name)
    with open(os.path.join(rd, 'eval_history.json'), 'w') as f:
        json.dump(eval_history, f, indent=2)

    return agent, eval_history


# ─────────────────────────────────────────────────────
# 4. EXPERIMENT DEFINITIONS
# ─────────────────────────────────────────────────────

def build_schedules(train_steps, constant_epsilons):
    """Build all epsilon schedules to test.

    Returns list of (label, schedule) tuples.
    """
    schedules = []

    # No decay: constant epsilon throughout
    for eps in constant_epsilons:
        schedules.append((
            f'const_eps{eps}',
            NoDecaySchedule(eps),
        ))

    # Linear decay (current default): 1.0 → 0.01 over first half
    schedules.append((
        'linear_default',
        LinearDecaySchedule(
            eps_start=1.0, eps_end=0.01,
            decay_fraction=0.5),
    ))

    # Linear decay variants
    schedules.append((
        'linear_slow',
        LinearDecaySchedule(
            eps_start=1.0, eps_end=0.01,
            decay_fraction=0.9),
    ))

    schedules.append((
        'linear_fast',
        LinearDecaySchedule(
            eps_start=1.0, eps_end=0.01,
            decay_fraction=0.2),
    ))

    # Exponential decay: reach 0.01 at half of training
    schedules.append((
        'exp_default',
        ExponentialDecaySchedule.from_target(
            eps_start=1.0, eps_end=0.01,
            target_steps=train_steps // 2),
    ))

    # Exponential decay: slower (reach 0.01 at 90% of training)
    schedules.append((
        'exp_slow',
        ExponentialDecaySchedule.from_target(
            eps_start=1.0, eps_end=0.01,
            target_steps=int(train_steps * 0.9)),
    ))

    # Exponential decay: faster (reach 0.01 at 20% of training)
    schedules.append((
        'exp_fast',
        ExponentialDecaySchedule.from_target(
            eps_start=1.0, eps_end=0.01,
            target_steps=int(train_steps * 0.2)),
    ))

    return schedules


def run_epsilon_experiments(args):
    """Run all epsilon schedule experiments."""
    train_steps = args.train_steps
    lam = args.lam
    traffic = args.traffic
    output_dir = args.output_dir

    constant_epsilons = args.epsilons

    schedules = build_schedules(train_steps, constant_epsilons)

    agent_types = (
        ['q_learning', 'sarsa'] if args.agent == 'both'
        else [args.agent])

    total_runs = len(schedules) * len(agent_types)

    print("=" * 112)
    print("EPSILON DECAY ABLATION STUDY")
    print("=" * 112)
    print(f"\n  Traffic: {traffic}, λ={lam}")
    print(f"  Train steps: {train_steps}")
    print(f"  Agent types: {agent_types}")
    print(f"  Schedules ({len(schedules)}):")
    for label, sched in schedules:
        print(f"    • {label}: {sched.to_dict()}")
    print(f"\n  Total runs: {total_runs}")
    print()

    all_results = {}
    all_paths = []
    total_t0 = time.time()

    run_idx = 0
    for agent_type in agent_types:
        agent_label = ('Q-learning' if agent_type == 'q_learning'
                       else 'SARSA')

        for sched_label, schedule in schedules:
            run_idx += 1
            run_name = (
                f"eps_{agent_label}_{sched_label}"
                f"_{traffic}_lam{lam}")

            print(f"\n{'━' * 112}")
            print(f"  [{run_idx}/{total_runs}] "
                  f"{agent_label} + {sched_label}")
            print(f"{'━' * 112}")

            cfg = TrainConfig(
                agent_type=agent_type,
                lam=lam,
                traffic=traffic,
                train_steps=train_steps,
                alpha=args.alpha,
                gamma=args.gamma,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay_steps=train_steps // 2,
                output_dir=output_dir,
                run_name=run_name,
                checkpoint_interval=args.checkpoint_interval,
                eval_interval=args.eval_interval,
                eval_steps=args.eval_steps,
                eval_seed=args.eval_seed,
                agent_seed=args.agent_seed,
                env_seed=args.env_seed,
                save_checkpoints=not args.no_checkpoints,
                reset_interval=args.reset_interval,
            )

            # Create agent with custom schedule
            common_kwargs = dict(
                alpha=cfg.alpha,
                gamma=cfg.gamma,
                q_init=cfg.q_init,
            )
            if agent_type == 'q_learning':
                agent = ScheduledQLearning(
                    schedule=schedule,
                    total_steps=train_steps,
                    **common_kwargs)
            else:
                agent = ScheduledSARSA(
                    schedule=schedule,
                    total_steps=train_steps,
                    **common_kwargs)

            agent.set_rng(
                np.random.RandomState(cfg.agent_seed))

            trained, history = train_scheduled_agent(
                agent, cfg, run_name)

            best_path = os.path.join(
                run_dir(cfg.output_dir, run_name), 'best.npz')
            all_paths.append(best_path)

            # Store final eval for summary
            if history:
                best_entry = max(
                    history, key=lambda e: e['avg_reward'])
                all_results[f"{agent_label}_{sched_label}"] = {
                    'path': best_path,
                    'best_reward': best_entry['avg_reward'],
                    'best_wait_i': best_entry['avg_wait_i'],
                    'best_step': best_entry['step'],
                    'schedule': sched_label,
                    'agent': agent_label,
                }

    # ── Final comparison ──
    print(f"\n\n{'=' * 112}")
    print("FINAL COMPARISON: All epsilon strategies")
    print(f"{'=' * 112}")

    # Run baselines
    results = {}

    rng1 = np.random.RandomState(args.eval_seed + 1)
    rp = RandomPolicy(rng1)
    results['Random'] = evaluate_policy(
        rp.select_action, args.eval_steps,
        lam, traffic, args.eval_seed)

    sp = SCANPolicy()
    results['SCAN'] = evaluate_policy(
        sp.select_action, args.eval_steps,
        lam, traffic, args.eval_seed)

    lp = LOOKPolicy()
    results['LOOK'] = evaluate_policy(
        lp.select_action, args.eval_steps,
        lam, traffic, args.eval_seed)

    rng2 = np.random.RandomState(args.eval_seed + 2)
    nc = NearestCallPolicy(rng2)
    results['Nearest'] = evaluate_policy(
        nc.select_action, args.eval_steps,
        lam, traffic, args.eval_seed)

    # Evaluate all trained agents
    from .utils.storage import load_checkpoint

    for label, info in all_results.items():
        path = info['path']
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping")
            continue

        q_table, metadata = load_checkpoint(path)
        agent_type_name = metadata.get(
            'agent_type', 'Q-learning')
        if agent_type_name == 'SARSA':
            eval_agent = SARSAAgent()
        else:
            eval_agent = QLearningAgent()
        eval_agent.load_q_table(q_table)
        eval_agent.set_rng(
            np.random.RandomState(args.eval_seed + 10))

        # Truncate label for display
        display = label[:16] if len(label) > 16 else label
        results[display] = evaluate_policy(
            lambda s, va, a=eval_agent: a.select_action(
                s, va, greedy=True),
            args.eval_steps, lam, traffic, args.eval_seed)

    print_results(results)

    # ── Summary table ──
    print(f"\n{'=' * 112}")
    print("SUMMARY: Best reward by schedule × agent")
    print(f"{'=' * 112}")
    print(f"\n{'Schedule':<25s} {'Agent':<12s} "
          f"{'Best Rew/s':>12s} {'Wait_i':>10s} "
          f"{'Best Step':>12s}")
    print('-' * 75)

    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['best_reward'],
        reverse=True)

    for label, info in sorted_results:
        print(f"{info['schedule']:<25s} "
              f"{info['agent']:<12s} "
              f"{info['best_reward']:>12.4f} "
              f"{info['best_wait_i']:>10.2f} "
              f"{info['best_step']:>12d}")

    total_time = time.time() - total_t0
    print(f"\n  Total time: {total_time:.1f}s "
          f"({total_time / 60:.1f} min)")

    # Save summary
    summary_dir = os.path.join(output_dir, '_epsilon_summary')
    _ensure_dir(summary_dir)
    summary = {
        'config': {
            'train_steps': train_steps,
            'lam': lam,
            'traffic': traffic,
            'agent_types': agent_types,
            'constant_epsilons': constant_epsilons,
        },
        'results': {k: {**v, 'path': str(v['path'])}
                    for k, v in all_results.items()},
        'total_time_seconds': total_time,
    }
    summary_path = os.path.join(
        summary_dir, f'epsilon_ablation_{traffic}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")
    print(f"  Model paths:")
    for p in all_paths:
        print(f"    {p}")

    return all_results


# ─────────────────────────────────────────────────────
# 5. CLI
# ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Epsilon decay ablation study')

    p.add_argument('--agent', default='both',
                   choices=['q_learning', 'sarsa', 'both'],
                   help='Agent type(s) to test')
    p.add_argument('--traffic', default='mixed',
                   choices=['uniform', 'up_peak',
                            'down_peak', 'mixed'],
                   help='Traffic pattern (default: mixed)')
    p.add_argument('--lam', type=float, default=0.001,
                   help='Arrival rate')
    p.add_argument('--train_steps', type=int,
                   default=5_000_000,
                   help='Training steps per run')
    p.add_argument('--epsilons', nargs='+', type=float,
                   default=[1.0, 0.5, 0.1, 0.01],
                   help='Constant epsilon values to test '
                        '(for no-decay schedule)')

    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--output_dir', default='results')
    p.add_argument('--checkpoint_interval', type=int,
                   default=500_000)
    p.add_argument('--eval_interval', type=int, default=200_000)
    p.add_argument('--eval_steps', type=int, default=10_000)
    p.add_argument('--eval_seed', type=int, default=42)
    p.add_argument('--agent_seed', type=int, default=456)
    p.add_argument('--env_seed', type=int, default=123)
    p.add_argument('--reset_interval', type=int, default=50_000)
    p.add_argument('--no_checkpoints', action='store_true',
                   help='Disable intermediate checkpoints')

    return p.parse_args()


def main():
    args = parse_args()
    run_epsilon_experiments(args)


if __name__ == '__main__':
    main()