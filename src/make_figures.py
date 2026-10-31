"""Generate all paper figures from the stored simulation results.

Reads results/*.json and writes PNG files to paper/figures/.
Usage: python src/make_figures.py
"""
import json
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from sim_core import cmds, errors, lm, make_layout, make_ranges, procrustes_align  # noqa: E402

warnings.filterwarnings("ignore")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
STYLE = {
    "cMDS": dict(color=BLUE, marker="o", ls="-"),
    "SMACOF": dict(color=ORANGE, marker="s", ls="-"),
    "LM": dict(color=AQUA, marker="^", ls="-"),
    "LM multi-start": dict(color=YELLOW, marker="D", ls="-"),
    "LM multi-start (soft-L1)": dict(color=MAGENTA, marker="v", ls="-", lw=2.4),
    "SDP": dict(color=GREEN, marker="P", ls="--"),
}
LABEL = {
    "cMDS": "cMDS", "SMACOF": "SMACOF", "LM": "NLS/LM",
    "LM multi-start": "NLS/LM multi-start",
    "LM multi-start (soft-L1)": "NLS/LM multi-start, soft-$\\ell_1$",
    "SDP": "SDP (ref.)",
}
METHODS = list(STYLE)
plt.rcParams.update({
    "font.size": 7, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 6.5,
    "axes.titlesize": 7.5, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "mathtext.default": "regular",
})
FAIL_M = 1.0
res = json.load(open(os.path.join(ROOT, "results", "results.json")))
ex2 = json.load(open(os.path.join(ROOT, "results", "grid_scatter.json")))


def collect(sweep, key, fn):
    xs = [row["tag"] for row in res[sweep]]
    out = {m: [fn(np.array([t[m][key] for t in row["trials"]])) for row in res[sweep]] for m in METHODS}
    return xs, out


def line(ax, xs, ys, scale=1.0, methods=METHODS):
    for m in methods:
        st = dict(STYLE[m])
        lw = st.pop("lw", 1.2)
        ax.plot(xs, np.array(ys[m]) * scale, lw=lw, ms=3.2, mec="white", mew=0.4,
                zorder=2 if lw > 2 else 3, label=LABEL[m], **st)


def nominor(ax):
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())


def title(ax, s):
    ax.set_title(s, loc="left")


def legend_row(fig, ax, extra=None, ncol=7, y=-0.02):
    h, l = ax.get_legend_handles_labels()
    if extra:
        h += extra[0]
        l += extra[1]
    fig.legend(h, l, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y), columnspacing=1.0,
               handletextpad=0.4)


fail = lambda v: float(np.mean(v > FAIL_M) * 100)  # noqa: E731

# =============================================================== F1 schematic
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25))
for ax in axes:
    ax.set_aspect("equal")
    ax.axis("off")
    ax.grid(False)
P = np.array([[0.0, 0.0], [2.0, 0.2], [2.6, 1.8], [1.0, 2.5], [-0.3, 1.5]])
lab = ["1", "2", "3", "4", "5"]

ax = axes[0]
for i in range(5):
    for j in range(i + 1, 5):
        ax.plot(P[[i, j], 0], P[[i, j], 1], color="#9a998f", lw=0.8, zorder=1)
ax.scatter(P[:, 0], P[:, 1], s=70, facecolor="white", edgecolor=BLUE, lw=1.5, zorder=3)
for i in range(5):
    ax.text(P[i, 0], P[i, 1], lab[i], ha="center", va="center", fontsize=6.5, zorder=4)
ax.text(1.0, -0.15, "$d_{12}$", color=ORANGE, ha="center", va="top", fontsize=7)
ax.text(2.45, 1.0, "$d_{23}$", color=ORANGE, ha="left", va="center", fontsize=7)
ax.text(1.3, 1.25, "$d_{ij}=\\|x_i-x_j\\|+n_{ij}+b_{ij}$", color=INK2, ha="center", fontsize=6.3,
        bbox=dict(fc="white", ec="none", pad=0.8))
ax.set_xlim(-0.8, 3.2)
ax.set_ylim(-0.8, 2.9)
ax.set_title("(a) range mesh", loc="left", fontsize=7.2)

ax = axes[1]
th = np.deg2rad(55)
R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
Q = (P - P.mean(0)) @ R.T + np.array([4.4, 1.1])
F = (P - P.mean(0)) * np.array([-1, 1]) + np.array([4.4, -1.9])
for pts, col, ls in ((P, BLUE, "-"), (Q, "#8c8b83", "-"), (F, "#8c8b83", "-")):
    order = [0, 1, 2, 3, 4, 0]
    ax.plot(pts[order, 0], pts[order, 1], color=col, lw=0.9, ls=ls)
    ax.scatter(pts[:, 0], pts[:, 1], s=16, facecolor="white", edgecolor=col, lw=1.0, zorder=3)
ax.add_patch(FancyArrowPatch((2.9, 1.3), (3.35, 1.3), arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.8))
ax.add_patch(FancyArrowPatch((2.9, 0.6), (3.35, -0.9), arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.8))
ax.text(2.6, 3.3, "rotation + translation", ha="left", fontsize=6.3, color=INK2)
ax.text(2.7, -1.0, "reflection", ha="right", fontsize=6.3, color=INK2)
ax.text(-0.3, -2.9, "identical distances $\\Rightarrow$\nframe must be fixed", ha="left", va="center", fontsize=6.3, color=INK2)
ax.set_xlim(-0.8, 6.2)
ax.set_ylim(-3.4, 3.7)
ax.set_title("(b) gauge freedom", loc="left", fontsize=7.2)

ax = axes[2]
A = np.array([[0.0, 0.0], [2.4, 0.0], [2.4, 1.6], [0.0, 1.6]])
E1, E2 = np.array([1.2, 2.9]), np.array([1.2, -1.3])
for i in range(4):
    for j in range(i + 1, 4):
        ax.plot(A[[i, j], 0], A[[i, j], 1], color="#9a998f", lw=0.7, zorder=1)
ax.plot([0, 2.4], [0.0, 0.0], color="#cfcec8", lw=0.5, zorder=0)
for E, col, ls in ((E1, BLUE, "-"), (E2, ORANGE, "--")):
    for a in (A[2], A[3]):
        ax.plot([E[0], a[0]], [E[1], a[1]], color=col, lw=0.9, ls=ls, zorder=1)
    ax.scatter(*E, s=40, facecolor="white", edgecolor=col, lw=1.4, zorder=3)
ax.scatter(A[:, 0], A[:, 1], s=26, facecolor="white", edgecolor=INK2, lw=1.1, zorder=3)
ax.plot([-0.4, 2.8], [0.8, 0.8], color=INK2, lw=0.6, ls=":")
ax.text(2.9, 0.8, "mirror", fontsize=6.0, color=INK2, va="center")
ax.text(1.2, 3.4, "true", ha="center", fontsize=6.3, color=BLUE)
ax.text(1.2, -1.85, "equally consistent flip", ha="center", fontsize=6.3, color=ORANGE)
ax.set_xlim(-0.7, 3.8)
ax.set_ylim(-3.3, 3.8)
ax.set_title("(c) flip ambiguity", loc="left", fontsize=7.2)
ax.text(1.2, -2.85, "2D analogue: node with 2 links flips\nacross their line (3D: 3 links, plane)", ha="center", va="center",
        fontsize=5.8, color=INK2)
fig.subplots_adjust(wspace=0.05)
fig.savefig(os.path.join(FIG, "f1_schematic.png"))
plt.close(fig)

# =============================================================== F3 example + covariance
sc = ex2["scatter"]
x = np.array(sc["x"])
est = np.array(sc["est"])
cov = np.array(sc["cov"])
n = len(x)
rng = np.random.default_rng(5)
D, W = make_ranges(x, 0.10, rng)
ests = {"cMDS": cmds(D, W)}
ests["LM"] = lm(D, W, ests["cMDS"])
K = 20
fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.1), gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
for ax, (i, j, xl, yl, ttl) in zip(axes[:2], [(0, 1, "x [m]", "y [m]", f"(a) top view, errors ×{K}"),
                                              (0, 2, "x [m]", "z [m]", f"(b) side view, errors ×{K}")]):
    for m, xe in ests.items():
        xa = procrustes_align(xe, x)
        xm = x + K * (xa - x)
        st = STYLE[m]
        for k in range(n):
            ax.plot([x[k, i], xm[k, i]], [x[k, j], xm[k, j]], color=st["color"], lw=0.8, zorder=3)
        ax.scatter(xm[:, i], xm[:, j], s=10, marker=st["marker"], color=st["color"], edgecolor="white",
                   lw=0.3, zorder=4, label=LABEL[m])
    ax.scatter(x[:, i], x[:, j], s=22, facecolor="white", edgecolor=INK, lw=0.9, zorder=5, label="true")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_aspect("equal", adjustable="datalim")
    title(ax, ttl)
# anchor with the largest vertical standard deviation
zstd = np.array([np.sqrt(cov[3 * k + 2, 3 * k + 2]) for k in range(n)])
k = int(np.argmax(zstd))
d = (est[:, k, :] - x[k]) * 100  # cm
inside = {}
for ax, (a, b, xl, yl, ttl, key) in zip(axes[2:], [(0, 1, "Δx [cm]", "Δy [cm]", f"(c) anchor {k + 1}: x–y", "xy"),
                                                  (0, 2, "Δx [cm]", "Δz [cm]", f"(d) anchor {k + 1}: x–z", "xz")]):
    ax.scatter(d[:, a], d[:, b], s=3, color=BLUE, alpha=0.35, lw=0, zorder=2)
    c2 = cov[np.ix_([3 * k + a, 3 * k + b], [3 * k + a, 3 * k + b])] * 1e4
    w, v = np.linalg.eigh(c2)
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    s95 = np.sqrt(5.991)
    ax.add_patch(Ellipse((0, 0), 2 * s95 * np.sqrt(w[1]), 2 * s95 * np.sqrt(w[0]), angle=ang, fill=False,
                         ec=ORANGE, lw=1.3, zorder=3))
    md = np.einsum("ni,ij,nj->n", d[:, [a, b]], np.linalg.inv(c2), d[:, [a, b]])
    inside[key] = float(np.mean(md <= 5.991) * 100)
    ax.axhline(0, color=GRID, lw=0.6)
    ax.axvline(0, color=GRID, lw=0.6)
    lim = 1.15 * max(np.abs(d[:, [a, b]]).max(), 1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    title(ax, ttl)
h, l = axes[0].get_legend_handles_labels()
h += [plt.Line2D([], [], color=BLUE, marker="o", ls="", ms=2.5, alpha=0.6),
      plt.Line2D([], [], color=ORANGE, lw=1.3)]
l += ["400 NLS/LM estimates", "95 % ellipse from $\\sigma^2(J^TJ)^+$"]
fig.subplots_adjust(wspace=0.45, bottom=0.2)
fig.legend(h, l, loc="upper center", ncol=6, bbox_to_anchor=(0.5, -0.02), columnspacing=1.0, handletextpad=0.4)
fig.savefig(os.path.join(FIG, "f3_example_cov.png"))
plt.close(fig)

# =============================================================== F4 noise / N / runtime
xs, med = collect("noise", "rmse", np.median)
crlb = [np.median([t["crlb"] for t in row["trials"]]) for row in res["noise"]]
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1))
ax = axes[0]
line(ax, np.array(xs) * 100, med, scale=100)
ax.plot(np.array(xs) * 100, np.array(crlb) * 100, color=INK2, ls=":", lw=1.2, label="Cramér–Rao bound")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([2, 5, 10, 20, 30])
ax.set_xticklabels(["2", "5", "10", "20", "30"])
ax.set_yticks([3, 5, 10, 20, 50])
ax.set_yticklabels(["3", "5", "10", "20", "50"])
nominor(ax)
ax.set_xlabel("range noise σ [cm]")
ax.set_ylabel("median anchor RMSE [cm]")
title(ax, "(a) accuracy vs. noise (N = 8)")
xn, medn = collect("nodes", "rmse", np.median)
crn = [np.median([t["crlb"] for t in row["trials"]]) for row in res["nodes"]]
ax = axes[1]
line(ax, xn, medn, scale=100)
ax.set_yscale("log")
ax.set_xticks(xn)
ax.set_yticks([10, 20, 30])
ax.set_yticklabels(["10", "20", "30"])
nominor(ax)
ax.set_xlabel("number of anchors N")
ax.set_ylabel("median anchor RMSE [cm]")
title(ax, "(b) accuracy vs. N (σ = 10 cm)")
_, tms = collect("nodes", "t", np.mean)
ax = axes[2]
line(ax, xn, tms, scale=1e3)
ax.set_yscale("log")
ax.set_xticks(xn)
nominor(ax)
ax.set_xlabel("number of anchors N")
ax.set_ylabel("mean runtime [ms]")
title(ax, "(c) runtime vs. N (desktop CPU)")
fig.subplots_adjust(wspace=0.32, bottom=0.27)
legend_row(fig, axes[0], ncol=7, y=-0.03)
fig.savefig(os.path.join(FIG, "f4_noise_nodes.png"))
plt.close(fig)

# =============================================================== F5 missing / NLOS
xm_, medm = collect("missing", "rmse", np.median)
_, flm = collect("missing", "rmse", fail)
xl_, medl = collect("nlos", "rmse", np.median)
_, fll = collect("nlos", "rmse", fail)
fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.1))
line(axes[0], np.array(xm_) * 100, medm, scale=100)
axes[0].set_yscale("log")
axes[0].set_yticks([10, 20, 50, 100, 300])
axes[0].set_yticklabels(["10", "20", "50", "100", "300"])
nominor(axes[0])
axes[0].set_xlabel("missing links [%]")
axes[0].set_ylabel("median RMSE [cm]")
title(axes[0], "(a) link loss")
line(axes[1], np.array(xm_) * 100, flm)
axes[1].set_xlabel("missing links [%]")
axes[1].set_ylabel("gross failures [%]")
title(axes[1], "(b) link loss")
line(axes[2], np.array(xl_) * 100, medl, scale=100)
axes[2].set_yscale("log")
axes[2].set_yticks([10, 20, 50, 100])
axes[2].set_yticklabels(["10", "20", "50", "100"])
nominor(axes[2])
axes[2].set_xlabel("NLOS-biased links [%]")
axes[2].set_ylabel("median RMSE [cm]")
title(axes[2], "(c) NLOS outliers")
line(axes[3], np.array(xl_) * 100, fll)
axes[3].set_xlabel("NLOS-biased links [%]")
axes[3].set_ylabel("gross failures [%]")
title(axes[3], "(d) NLOS outliers")
fig.subplots_adjust(wspace=0.42, bottom=0.27)
legend_row(fig, axes[0], ncol=6, y=-0.03)
fig.savefig(os.path.join(FIG, "f5_missing_nlos.png"))
plt.close(fig)

# =============================================================== F6 heatmaps
grid = ex2["grid"]
ns, miss = ex2["ns"], ex2["miss"]
cmap = LinearSegmentedColormap.from_list("blues", ["#eef5fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.95))
for ax, m, ttl in zip(axes, ["cMDS", "LM", "LM multi-start"],
                      ["(a) cMDS", "(b) NLS/LM, cMDS init", "(c) NLS/LM, 8 starts"]):
    Z = np.array([[np.mean(np.array(grid[f"{n_}|{mm}"][m]) > FAIL_M) * 100 for mm in miss] for n_ in ns])
    im = ax.imshow(Z, cmap=cmap, vmin=0, vmax=100, aspect="auto", origin="lower")
    for a in range(len(ns)):
        for b in range(len(miss)):
            ax.text(b, a, f"{Z[a, b]:.0f}", ha="center", va="center", fontsize=6.3,
                    color="white" if Z[a, b] > 45 else INK)
    ax.set_xticks(range(len(miss)))
    ax.set_xticklabels([f"{int(mm * 100)}" for mm in miss])
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels(ns)
    ax.set_xlabel("missing links [%]")
    ax.set_ylabel("anchors N")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    title(ax, ttl)
fig.subplots_adjust(wspace=0.35, right=0.9)
cax = fig.add_axes([0.92, 0.2, 0.012, 0.65])
cb = fig.colorbar(im, cax=cax)
cb.set_label("gross failures [%]", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
fig.savefig(os.path.join(FIG, "f6_heatmaps.png"))
plt.close(fig)

# =============================================================== height-sweep data (used by the height figure)
xh, medh = collect("height", "rmse", np.median)
_, ezh = collect("height", "ez", np.median)
_, exyh = collect("height", "exy", np.median)
crh = [np.median([t["crlb"] for t in row["trials"]]) * 100 for row in res["height"]]
mname = "LM multi-start (soft-L1)"

# =============================================================== F8 ECDF at 20 % link loss
row = [r for r in res["missing"] if abs(r["tag"] - 0.2) < 1e-9][0]
fig, ax = plt.subplots(figsize=(3.45, 1.85))
for m in METHODS:
    v = np.sort(np.array([t[m]["rmse"] for t in row["trials"]]) * 100)
    st = dict(STYLE[m])
    lw = st.pop("lw", 1.2)
    ax.step(v, np.arange(1, len(v) + 1) / len(v) * 100, where="post", color=st["color"], lw=lw,
            ls=st["ls"], label=LABEL[m], zorder=2 if lw > 2 else 3)
ax.set_xscale("log")
ax.axvline(100, color=INK2, ls=":", lw=0.9)
ax.text(104, 40, "1 m", color=INK2, fontsize=6.3)
ax.set_xticks([10, 20, 50, 100, 300, 1000])
ax.set_xticklabels(["10", "20", "50", "100", "300", "1000"])
nominor(ax)
ax.set_xlabel("anchor RMSE [cm]")
ax.set_ylabel("share of trials ≤ x [%]")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=5.8, columnspacing=0.8)
fig.savefig(os.path.join(FIG, "f8_ecdf.png"))
plt.close(fig)


# =============================================================== F9 cost vs robustness (Pareto)
iN = res["nodes"]
rowN10 = [r for r in iN if r["n"] == 10][0]
rt = {m: float(np.mean([t[m]["t"] for t in rowN10["trials"]]) * 1e3) for m in METHODS}
row20 = [r for r in res["missing"] if abs(r["tag"] - 0.2) < 1e-9][0]
f20 = {m: fail(np.array([t[m]["rmse"] for t in row20["trials"]])) for m in METHODS}
fig, ax = plt.subplots(figsize=(3.45, 1.95))
for m in METHODS:
    st = STYLE[m]
    ax.scatter(rt[m], f20[m], s=42, color=st["color"], marker=st["marker"], edgecolor="white", lw=0.5,
               zorder=3, label=LABEL[m])
ax.set_xscale("log")
ax.set_xlim(0.15, 600)
ax.set_ylim(-14, 112)
ax.set_xlabel("mean runtime at $N{=}10$ [ms]")
ax.set_ylabel("gross failures at 20 % loss [%]")
ax.legend(loc="upper right", fontsize=6.2)
ax.annotate("better", xy=(0.4, 40), xytext=(0.7, 70), fontsize=6.5, color=INK2,
            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
fig.savefig(os.path.join(FIG, "f9_pareto.png"))
plt.close(fig)

# =============================================================== F10 anatomy of a gross failure
fc = json.load(open(os.path.join(ROOT, "results", "failure_case.json")))
xt = np.array(fc["x"])
Wm = np.array(fc["W"])
sols = [np.array(s) for s in fc["sols"]]
n = len(xt)
xs_fail = procrustes_align(sols[0], xt)
xs_ok = procrustes_align(sols[2], xt)
fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.25), gridspec_kw={"width_ratios": [1, 1, 1, 0.95]})
for ax in axes[:3]:
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 21.5)
    ax.set_ylim(-1.5, 21.5)
    ax.set_xlabel("x [m]")
ax = axes[0]
for i in range(n):
    for j in range(i + 1, n):
        ax.plot(xt[[i, j], 0], xt[[i, j], 1], color="#9a998f" if Wm[i, j] else "#e3482f", lw=0.5 if Wm[i, j] else 0.7,
                ls="-" if Wm[i, j] else (0, (2, 2)), alpha=0.7 if Wm[i, j] else 0.9, zorder=1)
ax.scatter(xt[:, 0], xt[:, 1], s=22, facecolor="white", edgecolor=INK, lw=0.9, zorder=3)
ax.set_ylabel("y [m]")
title(ax, "(a) truth; 20 % links lost")
for ax, xe, col, ttl in ((axes[1], xs_fail, ORANGE, "(b) single start"),
                         (axes[2], xs_ok, YELLOW, "(c) best of 8 starts")):
    err = np.linalg.norm(xe - xt, axis=1)
    for k in range(n):
        ax.plot([xt[k, 0], xe[k, 0]], [xt[k, 1], xe[k, 1]], color=col, lw=1.0, zorder=2)
    ax.scatter(xt[:, 0], xt[:, 1], s=20, facecolor="white", edgecolor=INK, lw=0.9, zorder=3)
    ax.scatter(xe[:, 0], xe[:, 1], s=12, color=col, marker="o", edgecolor="white", lw=0.3, zorder=4)
    bad = err > 1.0
    ax.scatter(xe[bad, 0], xe[bad, 1], s=60, facecolor="none", edgecolor="#e3482f", lw=1.1, zorder=5)
    title(ax, ttl)
    ax.text(10, -0.6, "RMSE %.2f m" % errors(xe, xt)[0], ha="center", fontsize=6.5, color=INK)
ax = axes[3]
cst = np.array(fc["costs"])
cols = [ORANGE if r > 1 else YELLOW for r in fc["rmse"]]
labels = ["cMDS"] + [f"r{k}" for k in range(1, 8)]
ax.bar(range(8), cst, color=cols, edgecolor="white", lw=0.5)
ax.set_yscale("log")
ax.set_ylim(0.05, 8)
ax.grid(axis="x", visible=False)
ax.set_xticks(range(8))
ax.set_xticklabels(["c"] + [str(k) for k in range(1, 8)])
ax.set_xlabel("start (c = cMDS, 1–7 random)")
ax.set_ylabel("final cost [m²]")
ax.text(3.5, 1.9, "flipped\noptimum", color=ORANGE, fontsize=6.3, ha="left", va="center")
ax.text(7.4, 0.2, "correct optimum", color="#8a6000", fontsize=6.3, ha="right")
ax.axhline(0.1017, color=INK2, lw=0.6, ls=":")
title(ax, "(d) cost reveals the flip")
for a_ in axes:
    a_.set_box_aspect(1)
fig.subplots_adjust(wspace=0.42)
fig.savefig(os.path.join(FIG, "f10_failure.png"))
plt.close(fig)

# =============================================================== F11 height geometry
from grid_scatter import jacobian_fim  # noqa: E402
fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.1), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
for ax, hh in zip(axes[:3], (1.0, 4.0, 8.0)):
    rng2 = np.random.default_rng(21)
    xl = make_layout(8, (20.0, 20.0, hh), rng2)
    covz = np.linalg.pinv(jacobian_fim(xl, 0.10), rcond=1e-10)
    sz = np.array([np.sqrt(covz[3 * k + 2, 3 * k + 2]) for k in range(8)])
    for k in range(8):
        ax.plot([xl[k, 0]] * 2, [xl[k, 2] - sz[k], xl[k, 2] + sz[k]], color=ORANGE, lw=2.2, solid_capstyle="butt", zorder=2)
    ax.scatter(xl[:, 0], xl[:, 2], s=14, facecolor="white", edgecolor=INK, lw=0.9, zorder=3)
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1.3, 9.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    title(ax, f"({'abc'[(1.0, 4.0, 8.0).index(hh)]}) height spread {hh:g} m")
    ax.text(10, 8.3, "mean vertical σ (CRB): %.0f cm" % (sz.mean() * 100), ha="center", fontsize=6.2, color=ORANGE)
ax = axes[3]
ax.plot(xh, np.array(ezh[mname]) * 100, color=ORANGE, marker="s", ms=3.2, lw=1.4, mec="white", mew=0.4, label="vertical")
ax.plot(xh, np.array(exyh[mname]) * 100, color=BLUE, marker="o", ms=3.2, lw=1.4, mec="white", mew=0.4, label="horizontal")
ax.plot(xh[3:], crh[3:], color=INK2, ls=":", lw=1.2, label="CRB (total)")
ax.set_yscale("log")
ax.set_yticks([5, 10, 20, 50])
ax.set_yticklabels(["5", "10", "20", "50"])
ax.set_ylim(0.7, 70)
nominor(ax)
ax.set_xlabel("height spread [m]")
ax.set_ylabel("median RMSE [cm]")
ax.legend(loc="lower left")
title(ax, "(d) measured error")
for a_ in axes:
    a_.set_box_aspect(1)
fig.subplots_adjust(wspace=0.45)
fig.savefig(os.path.join(FIG, "f11_height_geometry.png"))
plt.close(fig)
print("figures written to", FIG)
