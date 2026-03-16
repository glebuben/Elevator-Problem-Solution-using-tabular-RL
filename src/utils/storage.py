"""Save / load Q-tables with full metadata."""

import json
import os
from pathlib import Path

import numpy as np


def _ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def run_dir(output_dir, run_name):
    return os.path.join(output_dir, run_name)


def checkpoint_dir(output_dir, run_name):
    return os.path.join(
        run_dir(output_dir, run_name), 'checkpoints')


def save_checkpoint(q_table, metadata, output_dir, run_name,
                    step=None, is_best=False):
    rd = run_dir(output_dir, run_name)
    _ensure_dir(rd)

    meta_bytes = np.frombuffer(
        json.dumps(metadata, indent=2).encode('utf-8'),
        dtype=np.uint8)

    if step is not None:
        cd = checkpoint_dir(output_dir, run_name)
        _ensure_dir(cd)
        path = os.path.join(cd, f'step_{step:08d}.npz')
        np.savez_compressed(
            path, q_table=q_table, metadata_json=meta_bytes)
        print(f"  [checkpoint] {path}")

    if is_best:
        path = os.path.join(rd, 'best.npz')
        np.savez_compressed(
            path, q_table=q_table, metadata_json=meta_bytes)
        print(f"  [best]       {path}")


def load_checkpoint(path):
    data = np.load(path, allow_pickle=False)
    q_table = data['q_table']
    meta_bytes = data['metadata_json'].tobytes()
    metadata = json.loads(meta_bytes.decode('utf-8'))
    return q_table, metadata


def list_checkpoints(output_dir, run_name):
    cd = checkpoint_dir(output_dir, run_name)
    if not os.path.isdir(cd):
        return []
    files = sorted(
        f for f in os.listdir(cd)
        if f.startswith('step_') and f.endswith('.npz'))
    return [os.path.join(cd, f) for f in files]


def save_config(config, output_dir, run_name):
    rd = run_dir(output_dir, run_name)
    _ensure_dir(rd)
    config.to_json(os.path.join(rd, 'config.json'))