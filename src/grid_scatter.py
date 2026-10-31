"""Failure-rate grid and covariance check.

1) Error grid over (N, missing-link fraction) for cMDS, NLS/LM and NLS/LM multi-start.
2) Monte Carlo scatter of one fixed constellation together with the Cramer-Rao covariance.

Usage: python src/grid_scatter.py [--trials 40] [--scatter 400] [--out results]
Output: results/grid_scatter.json
"""
import argparse
import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_core import (cmds, lm, lm_multistart, make_layout, make_ranges,  # noqa: E402
                      procrustes_align, errors)

warnings.filterwarnings("ignore")
BOX = (20.0, 20.0, 6.0)
SIGMA = 0.10
NS = (6, 8, 10, 14)
MISS = (0.0, 0.1, 0.2, 0.3, 0.4)
T = 40
N_SCATTER = 400


def grid_trial(args):
    n, miss, seed = args
    rng = np.random.default_rng(seed)
    x = make_layout(n, BOX, rng)
    D, W = make_ranges(x, SIGMA, rng, miss=miss)
    out = {}
    x0 = cmds(D, W)
    out["cMDS"] = errors(x0, x)[0]
    out["LM"] = errors(lm(D, W, x0), x)[0]
    out["LM multi-start"] = errors(lm_multistart(D, W), x)[0]
    return out


def jacobian_fim(x, sigma):
    n = len(x)
    j = np.zeros((3 * n, 3 * n))
    for a in range(n):
        for b in range(a + 1, n):
            u = x[a] - x[b]
            u /= np.linalg.norm(u)
            blk = np.outer(u, u) / sigma ** 2
            j[3 * a:3 * a + 3, 3 * a:3 * a + 3] += blk
            j[3 * b:3 * b + 3, 3 * b:3 * b + 3] += blk
            j[3 * a:3 * a + 3, 3 * b:3 * b + 3] -= blk
            j[3 * b:3 * b + 3, 3 * a:3 * a + 3] -= blk
    return j


def scatter_trial(seed_x):
    seed, x = seed_x
    rng = np.random.default_rng(seed)
    D, W = make_ranges(x, SIGMA, rng)
    xe = lm(D, W, cmds(D, W))
    return procrustes_align(xe, x)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=T, help="trials per (N, missing) cell")
    ap.add_argument("--scatter", type=int, default=N_SCATTER, help="repeated noisy meshes of the fixed layout")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    cli = ap.parse_args()
    T = cli.trials
    os.makedirs(cli.out, exist_ok=True)
    out = {"grid": {}, "ns": list(NS), "miss": list(MISS), "T": T}
    with Pool(processes=2) as pool:
        for n in NS:
            for m in MISS:
                args = [(n, m, 7000 + 131 * n + int(1000 * m) * 17 + i) for i in range(T)]
                res = pool.map(grid_trial, args)
                out["grid"][f"{n}|{m}"] = {k: [r[k] for r in res] for k in res[0]}
                print("grid", n, m, flush=True)
        # fixed constellation for the scatter figure
        rng = np.random.default_rng(11)
        x = make_layout(8, BOX, rng)
        est = pool.map(scatter_trial, [(90000 + i, x) for i in range(cli.scatter)])
    est = np.array(est)
    cov = np.linalg.pinv(jacobian_fim(x, SIGMA), rcond=1e-10)
    out["scatter"] = dict(x=x.tolist(), est=est.tolist(), cov=cov.tolist())
    json.dump(out, open(os.path.join(cli.out, "grid_scatter.json"), "w"))
    print("done", flush=True)
