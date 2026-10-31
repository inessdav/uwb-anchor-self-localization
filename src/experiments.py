"""Monte Carlo sweeps for range-only anchor self-localization.

Every trial draws a fresh random layout and range set; all methods see the
same data (paired comparison). Results are written to results/results.json.

Usage: python src/experiments.py [--trials 60] [--out results]
"""
import argparse
import json
import os
import sys
import time
import warnings
import zlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_core import (METHODS, crlb_rmse, make_layout, make_ranges,  # noqa: E402
                      procrustes_align, run_method)

warnings.filterwarnings("ignore")

BOX = (20.0, 20.0, 6.0)   # metres; flat-ish camp/indoor volume
SIGMA = 0.10              # m, typical DW3000 TWR range std (LOS, calibrated)
T = 60                    # default Monte Carlo trials per configuration


def one_trial(args):
    n, box, sigma, miss, out_frac, seed = args
    rng = np.random.default_rng(seed)
    x = make_layout(n, box, rng)
    D, W = make_ranges(x, sigma, rng, miss=miss, out_frac=out_frac)
    out = {"crlb": crlb_rmse(x, W, sigma) if out_frac == 0 else np.nan}
    for name in METHODS:
        rmse, emax, dt, xe = run_method(name, D, W, x)
        xa = procrustes_align(xe, x)
        ez = float(np.sqrt(np.mean((xa[:, 2] - x[:, 2]) ** 2)))
        exy = float(np.sqrt(np.mean(np.sum((xa[:, :2] - x[:, :2]) ** 2, axis=1))))
        out[name] = dict(rmse=rmse, emax=emax, t=dt, ez=ez, exy=exy)
    return out


def sweep(pool, label, configs):
    rows = []
    for cfg in configs:
        n, box, sigma, miss, out_frac, tag = cfg
        t0 = time.time()
        args = [(n, box, sigma, miss, out_frac, 1000 * hash_tag(label, tag) + i) for i in range(T)]
        res = pool.map(one_trial, args)
        rows.append(dict(tag=tag, n=n, box=list(box), sigma=sigma, miss=miss,
                         out_frac=out_frac, trials=res))
        print(f"[{label}] {tag}: {time.time() - t0:.1f}s", flush=True)
    return rows


def hash_tag(label, tag):
    return zlib.crc32(f"{label}|{tag}".encode()) % 9973 + 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=T, help="Monte Carlo trials per configuration")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    cli = ap.parse_args()
    T = cli.trials
    os.makedirs(cli.out, exist_ok=True)
    out = {}
    with Pool(processes=max(1, os.cpu_count() - 1)) as pool:
        out["noise"] = sweep(pool, "noise", [(8, BOX, s, 0.0, 0.0, s)
                                             for s in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)])
        out["nodes"] = sweep(pool, "nodes", [(n, BOX, SIGMA, 0.0, 0.0, n)
                                             for n in (4, 5, 6, 8, 10, 14, 20)])
        out["missing"] = sweep(pool, "missing", [(12, BOX, SIGMA, m, 0.0, m)
                                                 for m in (0.0, 0.1, 0.2, 0.3, 0.4)])
        out["height"] = sweep(pool, "height", [(8, (20.0, 20.0, h), SIGMA, 0.0, 0.0, h)
                                               for h in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)])
        out["nlos"] = sweep(pool, "nlos", [(12, BOX, SIGMA, 0.0, f, f)
                                           for f in (0.0, 0.05, 0.10, 0.20)])
    with open(os.path.join(cli.out, "results.json"), "w") as f:
        json.dump(out, f)
    print("done", flush=True)
