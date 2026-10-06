"""Train/evaluate the minimal model on cached simulator trajectories.

The final paper should extend this entry point with the selected high-fidelity
solver, physics projection, calibration split, and all preregistered metrics.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from flow_matching import FlowMatchingConfig, ConditionalVectorField, conditional_flow_matching_loss, sample_ode


def make_windows(states: np.ndarray, history: int, horizon: int):
    x, c = [], []
    for i in range(len(states) - history - horizon + 1):
        c.append(states[i : i + history].reshape(-1))
        x.append(states[i + history : i + history + horizon])
    return np.asarray(x, dtype=np.float32), np.asarray(c, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--history', type=int, default=1)
    p.add_argument('--horizon', type=int, default=32)
    p.add_argument('--out', type=Path, default=Path('results/model.pt'))
    args = p.parse_args()
    data = np.load(args.data)
    targets, conditions = make_windows(data['x'], args.history, args.horizon)
    # The condition encoder is deliberately compact for the first smoke run.
    conditions = conditions[:, -3:]
    ds = DataLoader(TensorDataset(torch.from_numpy(targets), torch.from_numpy(conditions)), batch_size=args.batch_size, shuffle=True)
    cfg = FlowMatchingConfig(state_dim=targets.shape[-1], horizon=args.horizon, condition_dim=conditions.shape[-1])
    model = ConditionalVectorField(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for target, condition in ds:
            opt.zero_grad(set_to_none=True)
            loss = conditional_flow_matching_loss(model, target, condition)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        print(f'epoch={epoch+1} loss={np.mean(losses):.6f}')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'config': cfg.__dict__, 'state_dict': model.state_dict()}, args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
