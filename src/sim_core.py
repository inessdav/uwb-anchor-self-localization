"""Core routines for UWB anchor self-localization experiments (3D, range-only).

Methods compared:
  cmds   - classical MDS (Torgerson), missing ranges filled by shortest paths
  smacof - weighted stress majorization (De Leeuw), initialized from cMDS
  lm     - nonlinear least squares (trust-region), initialized from cMDS
  lm_rob - same with a robust (soft-L1) loss
  sdp    - anchor-free SDP relaxation (offline reference), rank-3 projection
"""
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist


# ----------------------------------------------------------------------------
# Scenario generation
# ----------------------------------------------------------------------------
def make_layout(n, box, rng, min_sep=1.0):
    """Random anchor positions in a box (x, y, z extents in metres)."""
    box = np.asarray(box, dtype=float)
    for _ in range(1000):
        x = rng.uniform(0.0, 1.0, size=(n, 3)) * box
        d = cdist(x, x) + np.eye(n) * 1e9
        if d.min() >= min_sep:
            return x
    return x



def _is_rigid(x, W):
    """Infinitesimal rigidity test via rank of the rigidity matrix."""
    n = len(x)
    if np.ptp(x[:, 2]) < 0.05 or n < 4:   # (near-)coplanar: skip, degree test only
        return True
    iu = np.triu_indices(n, 1)
    sel = W[iu]
    ii, jj = iu[0][sel], iu[1][sel]
    r = np.zeros((len(ii), 3 * n))
    rows = np.arange(len(ii))
    diff = x[ii] - x[jj]
    for c in range(3):
        r[rows, 3 * ii + c] = diff[:, c]
        r[rows, 3 * jj + c] = -diff[:, c]
    return np.linalg.matrix_rank(r, tol=1e-8) == 3 * n - 6


def make_ranges(x, sigma, rng, miss=0.0, out_frac=0.0, out_bias=(0.5, 2.0)):
    """Simulate symmetric TWR range matrix.

    sigma    : std of Gaussian range noise [m]
    miss     : fraction of links removed (not measurable)
    out_frac : fraction of surviving links hit by a positive NLOS bias
    Returns D (n x n, symmetric) and boolean mask W (True = measured).
    """
    n = len(x)
    d_true = cdist(x, x)
    iu = np.triu_indices(n, 1)
    m = len(iu[0])
    for _ in range(200):
        keep = rng.uniform(size=m) >= miss
        W = np.zeros((n, n), dtype=bool)
        W[iu] = keep
        W = W | W.T
        # every node needs >= 4 neighbours and the framework must be
        # infinitesimally rigid (rank 3n-6) for a unique 3D realization
        if W.sum(axis=1).min() >= min(4, n - 1) and _is_rigid(x, W):
            break
    noise = rng.normal(0.0, sigma, size=m)
    bias = np.zeros(m)
    if out_frac > 0:
        hit = rng.uniform(size=m) < out_frac
        bias[hit] = rng.uniform(out_bias[0], out_bias[1], size=hit.sum())
    D = np.zeros((n, n))
    D[iu] = np.maximum(d_true[iu] + noise + bias, 1e-3)
    D = D + D.T
    return D, W


# ----------------------------------------------------------------------------
# Alignment / metrics
# ----------------------------------------------------------------------------
def procrustes_align(x_est, x_true):
    """Rigid alignment (rotation + translation + optional reflection)."""
    mu_e, mu_t = x_est.mean(0), x_true.mean(0)
    a, b = x_est - mu_e, x_true - mu_t
    u, _, vt = np.linalg.svd(a.T @ b)
    r = u @ vt
    return a @ r + mu_t


def errors(x_est, x_true):
    xa = procrustes_align(x_est, x_true)
    e = np.linalg.norm(xa - x_true, axis=1)
    return float(np.sqrt(np.mean(e ** 2))), float(e.max())


# ----------------------------------------------------------------------------
# Algorithms
# ----------------------------------------------------------------------------
def cmds(D, W, dim=3):
    n = len(D)
    g = csr_matrix(np.where(W, D, 0.0))
    dfill = shortest_path(g, directed=False)
    if not np.isfinite(dfill).all():
        dfill[~np.isfinite(dfill)] = dfill[np.isfinite(dfill)].max()
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (dfill ** 2) @ j
    w, v = np.linalg.eigh(b)
    idx = np.argsort(w)[::-1][:dim]
    return v[:, idx] * np.sqrt(np.maximum(w[idx], 0.0))


def smacof(D, W, x0, iters=500, tol=1e-9):
    n = len(D)
    wt = W.astype(float)
    np.fill_diagonal(wt, 0.0)
    v = np.diag(wt.sum(1)) - wt
    vp = np.linalg.pinv(v)
    x = x0.copy()
    prev = np.inf
    for k in range(iters):
        dz = cdist(x, x)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dz > 1e-12, wt * D / dz, 0.0)
        b = -ratio
        np.fill_diagonal(b, ratio.sum(1))
        x = vp @ b @ x
        stress = np.sum(wt * (cdist(x, x) - D) ** 2) / 2
        if prev - stress < tol * max(prev, 1.0):
            break
        prev = stress
    return x


def lm(D, W, x0, robust=False, f_scale=0.3):
    n = len(D)
    iu = np.triu_indices(n, 1)
    sel = W[iu]
    ii, jj = iu[0][sel], iu[1][sel]
    d = D[ii, jj]
    m = len(d)

    def res(p):
        x = p.reshape(n, 3)
        return np.linalg.norm(x[ii] - x[jj], axis=1) - d

    def jac(p):
        x = p.reshape(n, 3)
        diff = x[ii] - x[jj]
        nrm = np.maximum(np.linalg.norm(diff, axis=1, keepdims=True), 1e-9)
        u = diff / nrm
        jm = np.zeros((m, 3 * n))
        rows = np.arange(m)
        for c in range(3):
            jm[rows, 3 * ii + c] = u[:, c]
            jm[rows, 3 * jj + c] = -u[:, c]
        return jm

    kw = dict(loss="soft_l1", f_scale=f_scale) if robust else dict(loss="linear")
    sol = least_squares(res, x0.ravel(), jac=jac, method="trf", xtol=1e-10, ftol=1e-10, **kw)
    return sol.x.reshape(n, 3)


def sdp(D, W, lam=None):
    """Anchor-free SDP relaxation: Gram matrix G >= 0, centered, L1 fit of
    squared distances, with trace term pushing the embedding to full extent."""
    import cvxpy as cp

    n = len(D)
    g = cp.Variable((n, n), PSD=True)
    iu = np.triu_indices(n, 1)
    sel = W[iu]
    ii, jj = iu[0][sel], iu[1][sel]
    d2 = D[ii, jj] ** 2
    diag = cp.diag(g)
    dist2 = diag[ii] + diag[jj] - 2 * cp.hstack([g[a, b] for a, b in zip(ii, jj)])
    scale = float(np.mean(d2))
    if lam is None:
        lam = 1e-2 / n
    obj = cp.Minimize(cp.sum(cp.abs(dist2 - d2)) / scale - lam * cp.trace(g) / scale * n)
    prob = cp.Problem(obj, [g @ np.ones(n) == 0])
    prob.solve(solver=cp.CLARABEL)
    gv = g.value
    w, v = np.linalg.eigh((gv + gv.T) / 2)
    idx = np.argsort(w)[::-1][:3]
    return v[:, idx] * np.sqrt(np.maximum(w[idx], 0.0))


def _cost(x, D, W):
    n = len(D)
    iu = np.triu_indices(n, 1)
    sel = W[iu]
    r = np.linalg.norm(x[iu[0][sel]] - x[iu[1][sel]], axis=1) - D[iu][sel]
    return float(np.sum(r ** 2))


def lm_multistart(D, W, n_starts=8, robust=False, seed=0):
    """cMDS-initialized LM plus random restarts; keep the lowest-cost fit."""
    rng = np.random.default_rng(seed)
    x_c = cmds(D, W)
    scale = np.ptp(x_c, axis=0).max() + 1e-9
    starts = [x_c] + [rng.uniform(-0.5, 0.5, size=x_c.shape) * scale for _ in range(n_starts - 1)]
    best, best_c = None, np.inf
    for x0 in starts:
        x = lm(D, W, x0, robust=robust)
        c = _cost(x, D, W)
        if c < best_c:
            best, best_c = x, c
    return best


METHODS = {
    "cMDS": lambda D, W: cmds(D, W),
    "SMACOF": lambda D, W: smacof(D, W, cmds(D, W)),
    "LM": lambda D, W: lm(D, W, cmds(D, W)),
    "LM multi-start": lambda D, W: lm_multistart(D, W),
    "LM multi-start (soft-L1)": lambda D, W: lm_multistart(D, W, robust=True),
    "SDP": lambda D, W: sdp(D, W),
}


def run_method(name, D, W, x_true):
    t0 = time.perf_counter()
    x = METHODS[name](D, W)
    dt = time.perf_counter() - t0
    rmse, emax = errors(x, x_true)
    return rmse, emax, dt, x


# ----------------------------------------------------------------------------
# Fundamental limit (Cramer-Rao style) for anchor-free range-only estimation
# ----------------------------------------------------------------------------
def crlb_rmse(x_true, W, sigma):
    """sqrt(tr(J^+)/n): lower bound on gauge-invariant per-anchor RMSE.

    J is the Fisher information of Gaussian ranges (variance sigma^2); its six
    zero eigenvalues (rigid motions) are removed by the pseudo-inverse.
    """
    n = len(x_true)
    j = np.zeros((3 * n, 3 * n))
    for a in range(n):
        for b in range(a + 1, n):
            if not W[a, b]:
                continue
            u = x_true[a] - x_true[b]
            u = u / np.linalg.norm(u)
            blk = np.outer(u, u) / sigma ** 2
            j[3 * a:3 * a + 3, 3 * a:3 * a + 3] += blk
            j[3 * b:3 * b + 3, 3 * b:3 * b + 3] += blk
            j[3 * a:3 * a + 3, 3 * b:3 * b + 3] -= blk
            j[3 * b:3 * b + 3, 3 * a:3 * a + 3] -= blk
    jp = np.linalg.pinv(j, rcond=1e-10)
    return float(np.sqrt(np.trace(jp) / n))
