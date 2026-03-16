#!/usr/bin/env python3
"""Train Q-learning and/or SARSA agents.

Usage:
    python -m src.train
    python -m src.train --agent sarsa --lam 0.1 --traffic up_peak
"""

import argparse
import json
import time

import numpy as np

from .config import TrainConfig
from .env import ElevatorEnv
from .agents import QLearningAgent, SARSAAgent, TOTAL_STATES
from .env.constants import NUM_ACTIONS
from .utils.evaluation import evaluate_policy
from .utils.storage import save_checkpoint, save_config, run_dir


def make_agent(cfg: TrainConfig):
    cls = (QLearningAgent if cfg.agent_type == 'q_learning'
           else SARSAAgent)
    agent = cls(
        alpha=cfg.alpha,
        gamma=cfg.gamma,
        epsilon_start=cfg.epsilon_start,
        epsilon_end=cfg.epsilon_end,
        epsilon_decay_steps=cfg.epsilon_decay_steps,
        q_init=cfg.q_init,
    )
    agent.set_rng(np.random.RandomState(cfg.agent_seed))
    return agent


def train_agent(agent, cfg: TrainConfig):
    is_sarsa = isinstance(agent, SARSAAgent)
    rn = cfg.run_name or (
        f"{agent.name}_{cfg.traffic}_lam{cfg.lam}")

    save_config(cfg, cfg.output_dir, rn)

    print(f"\nTraining {agent.name}: {cfg.train_steps} steps")
    print(f"  λ={cfg.lam}, traffic={cfg.traffic}, "
          f"γ={agent.gamma}, α={agent.alpha}")
    print(f"  ε: {agent.epsilon_start} → {agent.epsilon_end} "
          f"over {agent.epsilon_decay_steps} steps")
    print(f"  Update: "
          f"{'r + γ Q(s′,a′) [on-policy]' if is_sarsa else 'r + γ max Q(s′,a′) [off-policy]'}")
    print(f"  States: {TOTAL_STATES}, Q-table: "
          f"{TOTAL_STATES * NUM_ACTIONS * 8 / 1024 / 1024:.1f} MB")
    print(f"  Output: {run_dir(cfg.output_dir, rn)}")

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
        agent.update(state, action, reward, next_state,
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
            }
            save_checkpoint(agent.Q, meta, cfg.output_dir, rn,
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
                }
                save_checkpoint(
                    agent.Q, meta, cfg.output_dir, rn,
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
            print(f"  Step {current_step:>8d} | "
                  f"ε={agent.epsilon:.4f} | "
                  f"rew/s={cr:.3f} | "
                  f"wait_i={m['avg_wait_i']:.2f} | "
                  f"served={m['served']:>4d}/"
                  f"{total_pax:>4d} | "
                  f"{elapsed:.1f}s{mark}")

    total_time = time.time() - t0
    print(f"\n{agent.name} done in {total_time:.1f}s")
    print(f"Best: step {best_step}, "
          f"avg_reward={best_reward:.4f}")

    rd = run_dir(cfg.output_dir, rn)
    with open(f"{rd}/eval_history.json", 'w') as f:
        json.dump(eval_history, f, indent=2)

    return agent, eval_history


def parse_args():
    p = argparse.ArgumentParser(
        description='Train elevator RL agents')
    p.add_argument('--agent', default='both',
                   choices=['q_learning', 'sarsa', 'both'])
    p.add_argument('--lam', type=float, default=0.05)
    p.add_argument('--traffic', default='uniform',
                   choices=['uniform', 'up_peak',
                            'down_peak', 'mixed'])
    p.add_argument('--train_steps', type=int, default=5_000_000)
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--eps_start', type=float, default=1.0)
    p.add_argument('--eps_end', type=float, default=0.01)
    p.add_argument('--output_dir', default='results')
    p.add_argument('--run_name', default=None)
    p.add_argument('--checkpoint_interval', type=int,
                   default=500_000)
    p.add_argument('--eval_interval', type=int, default=200_000)
    p.add_argument('--eval_steps', type=int, default=10_000)
    p.add_argument('--eval_seed', type=int, default=42)
    p.add_argument('--agent_seed', type=int, default=456)
    p.add_argument('--env_seed', type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    eps_decay = args.train_steps // 2

    agents_to_train = (
        ['q_learning', 'sarsa'] if args.agent == 'both'
        else [args.agent])

    for agent_type in agents_to_train:
        cfg = TrainConfig(
            agent_type=agent_type,
            lam=args.lam,
            traffic=args.traffic,
            train_steps=args.train_steps,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon_start=args.eps_start,
            epsilon_end=args.eps_end,
            epsilon_decay_steps=eps_decay,
            output_dir=args.output_dir,
            run_name=(
                args.run_name if args.agent != 'both'
                else f"{'Q-learning' if agent_type == 'q_learning' else 'SARSA'}"
                     f"_{args.traffic}_lam{args.lam}"),
            checkpoint_interval=args.checkpoint_interval,
            eval_interval=args.eval_interval,
            eval_steps=args.eval_steps,
            eval_seed=args.eval_seed,
            agent_seed=args.agent_seed,
            env_seed=args.env_seed,
        )
        agent = make_agent(cfg)
        train_agent(agent, cfg)


if __name__ == '__main__':
    main()