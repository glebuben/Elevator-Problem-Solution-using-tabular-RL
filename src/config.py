"""Central configuration for all training and evaluation runs."""

from dataclasses import dataclass, asdict
from typing import Optional
import json


@dataclass
class TrainConfig:
    # Environment
    lam: float = 0.05
    traffic: str = 'uniform'
    num_floors: int = 5
    capacity: int = 5

    # Agent
    agent_type: str = 'q_learning'
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 2_500_000
    q_init: float = -5.0

    # Training
    train_steps: int = 5_000_000
    eval_interval: int = 200_000
    eval_steps: int = 10_000
    reset_interval: int = 50_000
    eval_seed: int = 42
    agent_seed: int = 456
    env_seed: int = 123

    # Storage
    output_dir: str = 'results'
    run_name: Optional[str] = None
    save_checkpoints: bool = True
    checkpoint_interval: int = 500_000

    def to_dict(self):
        return asdict(self)

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})