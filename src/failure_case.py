"""Find a mesh where single-start NLS/LM fails but multi-start succeeds.

Searches seeds for N = 10 anchors and 20 % missing links, then stores the true layout,
the link mask, and the solutions and final costs of eight starts (cMDS start + seven
random starts). Output: results/failure_case.json
"""
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_core import _cost, cmds, errors, lm, lm_multistart, make_layout, make_ranges  # noqa: E402

warnings.filterwarnings("ignore")
BOX = (20.0, 20.0, 6.0)
N, MISS, SIGMA = 10, 0.2, 0.10
FAIL_RMSE, OK_RMSE = 1.0, 0.3   # metres


def build(seed):
    rng = np.random.default_rng(1000 + seed)
    x = make_layout(N, BOX, rng)
    D, W = make_ranges(x, SIGMA, rng, miss=MISS)
    return x, D, W


def find_seed(max_seed=400):
    for seed in range(max_seed):
        x, D, W = build(seed)
        if errors(lm(D, W, cmds(D, W)), x)[0] > FAIL_RMSE and errors(lm_multistart(D, W), x)[0] < OK_RMSE:
            return seed
    raise RuntimeError("no failure case found")


if __name__ == "__main__":
    seed = find_seed()
    x, D, W = build(seed)
    x_c = cmds(D, W)
    scale = np.ptp(x_c, axis=0).max() + 1e-9
    rs = np.random.default_rng(0)       # same random starts as lm_multistart(seed=0)
    starts = [x_c] + [rs.uniform(-0.5, 0.5, size=x_c.shape) * scale for _ in range(7)]
    sols = [lm(D, W, s) for s in starts]
    out = dict(x=x.tolist(), W=W.tolist(), sols=[s.tolist() for s in sols],
               costs=[_cost(s, D, W) for s in sols], rmse=[errors(s, x)[0] for s in sols])
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(out, open(os.path.join(out_dir, "failure_case.json"), "w"))
    print("seed", seed, "costs", np.round(out["costs"], 3), "rmse", np.round(out["rmse"], 2))
