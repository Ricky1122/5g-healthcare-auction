"""
Paper-style result figures for the LSTM-preference extension.

Reproduces the three result families from Kumar & Kumar:
  Fig. 2  Convergence of social welfare for step sizes Delta
  Fig. 3  Per-link demand-response gap G_wk = d_wk - r_kw
  Fig. 4  Evolution of BS bids  zeta_kw
  Fig. 5  Evolution of HSP bids varrho_wk

Uses paper utilities instantiated with LSTM-driven rho:
  S_wk = rho log(1 + h d),   T_kw = c r + (beta/2) r^2
  varrho_wk = d * dS/dd = rho h d / (1 + h d)
  zeta_kw   = (1/r) dT/dr = c/r + beta

Array layout in this repo is (BS, HSP); plot labels use paper names
HSP1 / HSP2 (random competing buyers), BS1--BS3.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_step

DEFAULT_PROCESSED = ROOT / "data" / "processed"
DEFAULT_FIGDIR = ROOT / "artifacts" / "figures"
EPS = 1e-12
DELTAS = (0.034, 0.040, 0.046)
FIG345_DELTA = 0.040
FIG3_ITERS = 300  # visual crop only; trajectories are unchanged
AXIS_ITERS = 80
R_BID_FLOOR = 1e-3  # Mbps; zeta = c/r + beta is singular at r = 0
LINESTYLES = ("-", "--", "-.", ":")


def _hsp(j: int) -> str:
    return f"HSP{int(j) + 1}"


def _bs(i: int) -> str:
    return f"BS{i + 1}"


def _axes_grid(
    n: int,
    max_cols: int,
    cell_w: float,
    cell_h: float,
    sharex: bool = True,
    sharey: bool = True,
):
    n = max(int(n), 1)
    cols = min(max(int(max_cols), 1), n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cell_w * cols, cell_h * rows),
        sharex=sharex,
        sharey=sharey,
    )
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
        ax.set_axis_off()
    return fig, axes[:n]

DELTA_STYLE = {
    0.004: dict(color="#2ca02c", marker="+", linestyle="-", label=r"$\Delta=0.004$"),
    0.006: dict(color="#8c564b", marker="v", linestyle="--", label=r"$\Delta=0.006$"),
    0.008: dict(color="#1f77b4", marker="o", linestyle="-.", label=r"$\Delta=0.008$"),
    0.01: dict(color="#2ca02c", marker="+", linestyle="-", label=r"$\Delta=0.01$"),
    0.012: dict(color="#2ca02c", marker="+", linestyle="-", label=r"$\Delta=0.012$"),
    0.015: dict(color="#2ca02c", marker="+", linestyle="-", label=r"$\Delta=0.015$"),
    0.018: dict(color="#2ca02c", marker="+", linestyle="-", label=r"$\Delta=0.018$"),
    0.025: dict(color="#8c564b", marker="v", linestyle="--", label=r"$\Delta=0.025$"),
    0.028: dict(color="#2ca02c", marker="+", linestyle="-", label=r"$\Delta=0.028$"),
    0.032: dict(color="#8c564b", marker="v", linestyle="--", label=r"$\Delta=0.032$"),
    0.034: dict(color="#2ca02c", marker="+", linestyle="-", label=r"$\Delta=0.034$"),
    0.035: dict(color="#8c564b", marker="v", linestyle="--", label=r"$\Delta=0.035$"),
    0.038: dict(color="#8c564b", marker="v", linestyle="--", label=r"$\Delta=0.038$"),
    0.04: dict(color="#1f77b4", marker="o", linestyle="-.", label=r"$\Delta=0.04$"),
    0.046: dict(color="#1f77b4", marker="o", linestyle="-.", label=r"$\Delta=0.046$"),
    0.05: dict(color="#1f77b4", marker="o", linestyle="-.", label=r"$\Delta=0.050$"),
    0.055: dict(color="#1f77b4", marker="o", linestyle="-.", label=r"$\Delta=0.055$"),
    0.02: dict(color="#8c564b", marker="v", linestyle="--", label=r"$\Delta=0.02$"),
    0.06: dict(color="#1f77b4", marker="o", linestyle="-.", label=r"$\Delta=0.06$"),
    0.07: dict(color="#1f77b4", marker="o", linestyle="-.", label=r"$\Delta=0.070$"),
    0.075: dict(color="#1f77b4", marker="o", linestyle="-.", label=r"$\Delta=0.075$"),
    0.08: dict(color="#ff7f0e", marker="s", linestyle=":", label=r"$\Delta=0.08$"),
    0.085: dict(color="#1f77b4", marker="o", linestyle="-.", label=r"$\Delta=0.085$"),
    0.09: dict(color="#1f77b4", marker="o", linestyle="-.", label=r"$\Delta=0.090$"),
}

_FALLBACK_DELTA = (
    dict(color="#2ca02c", marker="+", linestyle="-"),
    dict(color="#8c564b", marker="v", linestyle="--"),
    dict(color="#1f77b4", marker="o", linestyle="-."),
)
BS_STYLE = (
    dict(color="#2ca02c", marker="+", linestyle="-"),
    dict(color="#d62728", marker="x", linestyle="--"),
    dict(color="#1f77b4", marker="o", linestyle="-."),
    dict(color="#9467bd", marker="s", linestyle=":"),
    dict(color="#ff7f0e", marker="v", linestyle="-"),
)
HSP_STYLE = (
    dict(color="#2ca02c", marker="+", linestyle="-"),
    dict(color="#1f77b4", marker="o", linestyle="--"),
    dict(color="#d62728", marker="x", linestyle="-."),
    dict(color="#9467bd", marker="s", linestyle=":"),
    dict(color="#ff7f0e", marker="v", linestyle="-"),
)


def _mark_period() -> int:
    """Keep about 10 markers on the axis, the same density as the old 80-iter / every-8 plots."""
    return max(8, int(round(AXIS_ITERS / 10)))


def _trace_style(base: dict, index: int) -> dict:
    style = dict(base)
    style.setdefault("linestyle", LINESTYLES[index % len(LINESTYLES)])
    style["markevery"] = (2 + 3 * index, _mark_period())
    style.setdefault("markersize", 4.2)
    style["markeredgewidth"] = 0.85
    style.setdefault("linewidth", 1.45)
    style["zorder"] = 3 + index
    return style


def _nice_end(raw: int, pad: int = 0) -> int:
    raw = int(raw) + int(pad)
    step = 50 if raw >= 200 else 20 if raw >= 100 else 10
    return int(max(step, np.ceil(raw / step) * step))


def _axis_end(traj: dict[float, dict], tol: float = 1e-3, pad: int = 30) -> int:
    """Show the full rise through lock-in, then stop so the tail does not flatten the story."""
    hits = []
    for tr in traj.values():
        ok = np.where(np.asarray(tr["gap"]) < tol)[0]
        hits.append(int(ok[0]) + 1 if len(ok) else int(tr["n"]))
    return _nice_end(max(hits), pad)


def _sw_axis_end(
    traj: dict[float, dict], max_sw: float | None = None, frac: float = 0.997, pad: int = 40
) -> int:
    """Crop Fig. 2 to when welfare actually arrives, not to the later gap tail."""
    finals = [float(tr["sw_matched"][-1]) for tr in traj.values()]
    ref = max([x for x in (*finals, max_sw or 0.0) if x is not None] or [1.0])
    hits = []
    for tr in traj.values():
        sw = np.asarray(tr["sw_matched"], dtype=np.float64)
        target = frac * max(ref, float(sw[-1]))
        ok = np.where(sw >= target)[0]
        hits.append(int(ok[0]) + 1 if len(ok) else int(tr["n"]))
    return _nice_end(max(hits), pad)


def _tick_step(xmax: int) -> int:
    if xmax <= 80:
        return 10
    if xmax <= 160:
        return 20
    if xmax <= 300:
        return 25
    return 50


def _iter_axis(ax, xmax: int | None = None, tick: int | None = None) -> None:
    xmax = int(AXIS_ITERS if xmax is None else xmax)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_locator(MultipleLocator(tick if tick is not None else _tick_step(xmax)))


def _legend(ax, loc: str = "upper right", **kwargs) -> None:
    opts = dict(fontsize=7, framealpha=0.95, handlelength=2.4, borderpad=0.3)
    opts.update(kwargs)
    ax.legend(loc=loc, **opts)


def _figure_legend(fig, ax, ncol: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    for a in fig.axes:
        if a.get_legend() is not None:
            a.get_legend().remove()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        fontsize=7,
        framealpha=0.95,
        handlelength=2.4,
        bbox_to_anchor=(0.5, 1.03),
    )


def _pick_trace(traj: dict[float, dict]) -> dict:
    """Use the mid/small step for Figs 3–5. Smallest terminal gap is often the largest Δ."""
    if FIG345_DELTA in traj:
        return traj[FIG345_DELTA]
    return traj[min(traj.keys())]


def _style_for_delta(delta: float) -> dict:
    key = round(float(delta), 3)
    if key in DELTA_STYLE:
        return dict(DELTA_STYLE[key])
    style = dict(_FALLBACK_DELTA[hash(key) % len(_FALLBACK_DELTA)])
    style["label"] = rf"$\Delta={delta:g}$"
    return style


def _ieee_rc() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.45,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "lines.linewidth": 1.45,
            "lines.markersize": 4.2,
        }
    )


def _load_market(processed_dir: Path, opt, omega_path: Path | None = None):
    z = np.load(processed_dir / "rho_wk.npz", allow_pickle=True)
    rho = np.asarray(z["rho_wk"], dtype=np.float64)
    n_wk = np.asarray(z["N_wk"], dtype=np.int64)
    gain = np.asarray(z["channel_gain_wk"], dtype=np.float64)
    demand_k = np.asarray(z["demand_k"], dtype=np.float64)
    hsp_names = [str(x) for x in z["hsp_names"]]
    active = n_wk > 0
    cfg = opt.OptimizerConfig(
        processed_dir=processed_dir,
        omega_path=omega_path,
        bs_capacity_mbps=5.0,
        scale_capacity_by_load=False,
    )
    h = opt.normalize_snr(gain, cfg.snr_scale)
    h = np.where(active, h, EPS)
    n_bs = rho.shape[0]
    r_w = np.full(n_bs, float(cfg.bs_capacity_mbps), dtype=np.float64)
    d_k = cfg.oversubscribe * float(r_w.sum()) * (demand_k / max(float(demand_k.sum()), EPS))
    omega = opt.load_omega(cfg.omega_path, rho.shape)
    cost = opt._as_matrix(cfg.cost_c, rho.shape) * omega
    beta = opt._as_matrix(cfg.congestion_beta, rho.shape) * omega
    return {
        "rho": rho,
        "h": h,
        "active": active,
        "R_w": r_w,
        "D_k": d_k,
        "cost": cost,
        "beta": beta,
        "omega": omega,
        "hsp_names": hsp_names,
        "cfg": cfg,
    }


def hsp_bid(rho: np.ndarray, h: np.ndarray, d: np.ndarray) -> np.ndarray:
    """varrho_wk = d * dS/dd  (paper Eq. 21) for S = rho log(1+h d)."""
    return rho * h * d / np.maximum(1.0 + h * d, EPS)


def bs_bid(
    cost: np.ndarray,
    beta: np.ndarray,
    r: np.ndarray,
    omega: np.ndarray | None = None,
) -> np.ndarray:
    """zeta_kw = (1/r) dT/dr. If cost/beta are unscaled, pass omega to apply T=ω g(r)."""
    z = cost / np.maximum(r, R_BID_FLOOR) + beta
    if omega is None:
        return z
    return np.asarray(omega, dtype=np.float64) * z


def run_trajectory(
    mkt: dict, opt, delta: float, max_iter: int, seed: int, pi_init: float | None = None
) -> dict:
    """Constant-step sub-gradient on pi; log d, r, SW, bids every iteration."""
    rho, h, active = mkt["rho"], mkt["h"], mkt["active"]
    cfg = mkt["cfg"]
    n_bs, n_hsp = rho.shape
    rng = np.random.default_rng(seed)
    pi_floor = max(cfg.pi_min, 0.5 * cfg.cost_c)
    start = cfg.pi_init if getattr(cfg, "pi_init", None) is not None else pi_init
    if start is not None:
        pi = np.full(rho.shape, float(start), dtype=np.float64) + 0.01 * rng.random(rho.shape)
    else:
        pi = cfg.cost_c + 0.04 * np.maximum(rho, 0.0) + 0.005 * rng.random(rho.shape)
    pi = np.clip(pi, pi_floor, cfg.pi_max)
    pi = np.where(active, pi, pi_floor)

    rec = {
        "d": np.zeros((max_iter, n_bs, n_hsp)),
        "r": np.zeros((max_iter, n_bs, n_hsp)),
        "sw": np.zeros(max_iter),
        "sw_matched": np.zeros(max_iter),
        "gap": np.zeros(max_iter),
        "varrho": np.zeros((max_iter, n_bs, n_hsp)),
        "zeta": np.zeros((max_iter, n_bs, n_hsp)),
    }
    n_done = 0
    for t in range(max_iter):
        d, _ = opt.solve_opt1(rho, h, pi, mkt["D_k"], active, cfg.bisect_iters)
        r, _ = opt.solve_opt2(pi, mkt["cost"], mkt["beta"], mkt["R_w"], active, cfg.bisect_iters)
        matched = np.minimum(d, r)
        rec["d"][t] = d
        rec["r"][t] = r
        rec["sw"][t] = opt.welfare(d, r, rho, h, mkt["cost"], mkt["beta"])
        rec["sw_matched"][t] = opt.welfare(matched, matched, rho, h, mkt["cost"], mkt["beta"])
        rec["gap"][t] = opt.market_gap(d, r)
        rec["varrho"][t] = hsp_bid(rho, h, d)
        rec["zeta"][t] = bs_bid(mkt["cost"], mkt["beta"], r)
        n_done = t + 1
        excess = d - r
        step = delta / np.sqrt(t + 1.0)
        pi = np.clip(pi + step * excess / (1.0 + np.abs(excess)), pi_floor, cfg.pi_max)
        pi = np.where(active, pi, pi_floor)

    for key, arr in rec.items():
        rec[key] = arr[:n_done]
    rec["n"] = n_done
    rec["delta"] = delta
    return rec


def _save(fig, figdir: Path, stem: str) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figdir / f"{stem}.png")
    plt.close(fig)


def _official_welfare(processed_dir: Path | None) -> float | None:
    if processed_dir is None:
        return None
    path = Path(processed_dir) / "optimizer_summary.json"
    if not path.exists():
        return None
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))["welfare"])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def fig_convergence(
    traj: dict[float, dict], figdir: Path, max_sw: float | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    if max_sw is None:
        max_sw = max(float(tr["sw_matched"][-1]) for tr in traj.values())
    ax.axhline(
        max_sw,
        color="#d62728",
        linestyle=":",
        linewidth=1.2,
        label="Maximum Social welfare",
    )
    ys = [max_sw]
    for i, delta in enumerate(traj):
        tr = traj[delta]
        it = np.arange(1, tr["n"] + 1)
        ax.plot(it, tr["sw_matched"], **_trace_style(_style_for_delta(delta), i))
        ys.append(tr["sw_matched"])
    stacked = np.concatenate([np.atleast_1d(y) for y in ys])
    lo, hi = float(np.min(stacked)), float(np.max(stacked))
    span = max(hi - lo, 0.25)
    ax.set_ylim(lo - 0.08 * span, hi + 0.10 * span)
    ax.set_xlabel("iteration")
    ax.set_ylabel("Social welfare")
    _iter_axis(ax)
    _legend(ax, loc="lower right")
    fig.tight_layout()
    _save(fig, figdir, "fig2_convergence")


def _gap_trace_style(index: int) -> dict:
    style = _trace_style(BS_STYLE[index % len(BS_STYLE)], index)
    style["markevery"] = (12, 25)
    style["markersize"] = 5.0
    return style


def _gap_ylim(series: list, n_show: int) -> tuple[float, float]:
    """Zoom onto the approach to zero; one-sample cold-start spikes stay off-axis."""
    body = np.concatenate([np.asarray(g, dtype=np.float64)[5:n_show] for g in series])
    lo, hi = float(np.min(body)), float(np.max(body))
    span = max(hi - lo, 0.25)
    return lo - 0.12 * span, hi + 0.16 * span


def _gap_show_len(tr: dict) -> int:
    n_show = min(int(FIG3_ITERS), int(tr["n"]))
    n_show = max(n_show, min(int(AXIS_ITERS), int(tr["n"])))
    return max(n_show, 20)


def fig_gap(traj: dict, figdir: Path, n_hsp_panels: int = 2) -> None:
    tr = _pick_trace(traj)
    n_bs, n_hsp = tr["d"].shape[1], tr["d"].shape[2]
    n_show = _gap_show_len(tr)
    it = np.arange(1, tr["n"] + 1)
    all_gaps = [tr["d"][:, i, j] - tr["r"][:, i, j] for j in range(n_hsp) for i in range(n_bs)]
    ylim = _gap_ylim(all_gaps, n_show)
    tick = 50 if n_show >= 150 else None

    if n_hsp <= 3:
        n_hsp_panels = min(n_hsp_panels, n_hsp)
        fig, axes = plt.subplots(1, n_hsp_panels, figsize=(6.8, 3.8), sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for j, ax in enumerate(axes):
            for i in range(n_bs):
                g = tr["d"][:, i, j] - tr["r"][:, i, j]
                ax.plot(
                    it,
                    g,
                    label=f"{_hsp(j)}-{_bs(i)}",
                    **_gap_trace_style(i),
                )
            ax.axhline(0.0, color="k", linewidth=0.7, alpha=0.55)
            ax.set_xlabel("iteration")
            ax.set_ylim(*ylim)
            _iter_axis(ax, xmax=n_show, tick=tick)
            _legend(ax, loc="lower right", fontsize=6.5)
        axes[0].set_ylabel("Demand and response gap")
        fig.tight_layout()
        _save(fig, figdir, "fig3_demand_response_gap")

    max_cols = 3 if n_hsp <= 6 else 5
    fig, axes = _axes_grid(n_hsp, max_cols, 3.35, 3.15)
    for j, ax in enumerate(axes):
        for i in range(n_bs):
            g = tr["d"][:, i, j] - tr["r"][:, i, j]
            ax.plot(
                it,
                g,
                label=_bs(i),
                **_gap_trace_style(i),
            )
        ax.axhline(0.0, color="k", linewidth=0.7, alpha=0.55)
        ax.set_xlabel("iteration")
        ax.set_title(_hsp(j), fontsize=9)
        ax.set_ylim(*ylim)
        _iter_axis(ax, xmax=n_show, tick=tick)
    axes[0].set_ylabel("Demand and response gap")
    _figure_legend(fig, axes[0], ncol=min(n_bs, 5))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    _save(fig, figdir, "fig3_demand_response_gap_all")


def fig_bs_bids(traj: dict, figdir: Path) -> None:
    tr = _pick_trace(traj)
    n_bs, n_hsp = tr["zeta"].shape[1], tr["zeta"].shape[2]
    it = np.arange(1, tr["n"] + 1)
    max_cols = 3 if n_bs >= 3 else n_bs
    fig, axes = _axes_grid(n_bs, max_cols, 3.2, 3.35, sharey=False)
    named = n_hsp <= 4
    cmap = plt.cm.viridis
    for i, ax in enumerate(axes):
        for j in range(n_hsp):
            if named:
                style = _trace_style(HSP_STYLE[j % len(HSP_STYLE)], j)
                style["label"] = _hsp(j)
            else:
                style = _trace_style(HSP_STYLE[j % len(HSP_STYLE)], j)
                style["color"] = cmap(j / max(n_hsp - 1, 1))
                style["label"] = None
            ax.plot(it, tr["zeta"][:, i, j], **style)
        ax.set_title(_bs(i), fontsize=9)
        ax.set_ylabel("BSs Bids")
        ax.set_xlabel("iteration")
        _iter_axis(ax)
    if named:
        _figure_legend(fig, axes[0], ncol=n_hsp)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    else:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, n_hsp))
        sm.set_array([])
        fig.subplots_adjust(right=0.90, top=0.90, wspace=0.28, hspace=0.38)
        fig.colorbar(sm, ax=list(axes), fraction=0.03, pad=0.04, label="HSP index")
    _save(fig, figdir, "fig4_bs_bids")


def fig_hsp_bids(traj: dict, figdir: Path) -> None:
    tr = _pick_trace(traj)
    n_bs, n_hsp = tr["varrho"].shape[1], tr["varrho"].shape[2]
    it = np.arange(1, tr["n"] + 1)
    max_cols = 3 if n_hsp <= 6 else 5
    fig, axes = _axes_grid(n_hsp, max_cols, 3.2, 3.2, sharey=False)
    for j, ax in enumerate(axes):
        for i in range(n_bs):
            ax.plot(
                it,
                tr["varrho"][:, i, j],
                label=_bs(i),
                **_trace_style(BS_STYLE[i % len(BS_STYLE)], i),
            )
        ax.set_title(_hsp(j), fontsize=9)
        ax.set_xlabel("iteration")
        _iter_axis(ax)
    axes[0].set_ylabel("HSPs Bids")
    _figure_legend(fig, axes[0], ncol=min(n_bs, 5))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    _save(fig, figdir, "fig5_hsp_bids")


def fig_omega_const(omega: np.ndarray, figdir: Path, hsp_names: list[str]) -> None:
    n_bs, n_hsp = omega.shape
    names = list(hsp_names) if hsp_names else [_hsp(j) for j in range(n_hsp)]
    if n_hsp * n_bs <= 15:
        x = np.arange(n_hsp)
        width = 0.8 / max(n_bs, 1)
        bar_colors = ("#2ca02c", "#d62728", "#1f77b4", "#9467bd", "#ff7f0e")
        fig, ax = plt.subplots(figsize=(max(5.2, 0.7 * n_hsp + 1.8), 3.6))
        for i in range(n_bs):
            ax.bar(
                x + (i - (n_bs - 1) / 2) * width,
                omega[i],
                width,
                label=_bs(i),
                color=bar_colors[i % len(bar_colors)],
                edgecolor="k",
                linewidth=0.4,
            )
        ax.set_xticks(x, names, rotation=0 if n_hsp <= 6 else 45, ha="center" if n_hsp <= 6 else "right")
        ax.set_xlabel("HSP")
        ax.set_ylabel(r"Reluctance $\omega_{kw}$ (constant)")
        ax.set_ylim(0.0, 1.0)
        ax.legend(fontsize=7, framealpha=0.92, ncol=min(n_bs, 3))
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(max(6.4, 0.52 * n_hsp + 1.8), max(3.4, 0.42 * n_bs + 1.4)))
        im = ax.imshow(omega, vmin=0.0, vmax=1.0, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(np.arange(n_hsp), names, rotation=45, ha="right")
        ax.set_yticks(np.arange(n_bs), [_bs(i) for i in range(n_bs)])
        fontsize = 6 if n_hsp >= 8 else 8
        for i in range(n_bs):
            for j in range(n_hsp):
                ax.text(j, i, f"{omega[i, j]:.2f}", ha="center", va="center", fontsize=fontsize)
        ax.set_xlabel("HSP")
        ax.set_ylabel("BS")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"Reluctance $\omega_{kw}$")
        fig.tight_layout()
    _save(fig, figdir, "fig_omega_const")


def persist_arrays(traj: dict[float, dict], figdir: Path, hsp_names: list[str]) -> None:
    payload = {
        "hsp_map": {name: name for name in hsp_names},
        "deltas": [float(x) for x in traj.keys()],
    }
    for delta, tr in traj.items():
        tag = f"{delta:.3f}".replace(".", "p")
        np.savez_compressed(
            figdir / f"trace_delta_{tag}.npz",
            d=tr["d"],
            r=tr["r"],
            sw_matched=tr["sw_matched"],
            sw=tr["sw"],
            gap=tr["gap"],
            varrho=tr["varrho"],
            zeta=tr["zeta"],
        )
        payload[str(delta)] = {
            "n_iter": int(tr["n"]),
            "final_sw": float(tr["sw_matched"][-1]),
            "final_gap": float(tr["gap"][-1]),
        }
    (figdir / "figure_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_figures(
    processed_dir: Path | None = None,
    figdir: Path | None = None,
    max_iter: int = 450,
    seed: int = 42,
    omega_path: Path | None = None,
    copy_to_report: bool = True,
    deltas: tuple[float, ...] | None = None,
    pi_init: float | None = None,
    fig345_delta: float | None = None,
) -> Path:
    processed_dir = Path(processed_dir or DEFAULT_PROCESSED)
    figdir = Path(figdir or DEFAULT_FIGDIR)
    figdir.mkdir(parents=True, exist_ok=True)
    for stale in figdir.glob("trace_delta_*.npz"):
        stale.unlink()
    opt = load_step("optimizer")
    mkt = _load_market(processed_dir, opt, omega_path=omega_path)
    print(
        f"[figures] market  BS x HSP = {mkt['rho'].shape}  "
        f"HSP={mkt['hsp_names']}  supply={mkt['R_w'].sum():.1f} Mbps"
    )
    used = tuple(deltas or DELTAS)
    traj = {}
    for delta in used:
        tr = run_trajectory(mkt, opt, delta=delta, max_iter=max_iter, seed=seed, pi_init=pi_init)
        traj[delta] = tr
        print(
            f"[figures] Delta={delta:.3f}  iters={tr['n']}  "
            f"SW={tr['sw_matched'][-1]:.3f}  gap={tr['gap'][-1]:.3e}"
        )
    global AXIS_ITERS, FIG345_DELTA
    preferred = float(fig345_delta) if fig345_delta is not None else float(FIG345_DELTA)
    if preferred in traj and float(traj[preferred]["gap"][-1]) < 1e-2:
        FIG345_DELTA = preferred
    else:
        FIG345_DELTA = min(traj, key=lambda d: float(traj[d]["gap"][-1]))
        if preferred != FIG345_DELTA:
            print(
                f"[figures] fig3-5 using Delta={FIG345_DELTA:g} "
                f"(preferred {preferred:g} still open)"
            )
    official = _official_welfare(processed_dir)
    sw_end = _sw_axis_end(traj, official)
    gap_end = _axis_end({FIG345_DELTA: traj[FIG345_DELTA]})
    _ieee_rc()
    AXIS_ITERS = sw_end
    print(f"[figures] fig2 axis 0-{AXIS_ITERS}  (markers every {_mark_period()})")
    fig_convergence(traj, figdir, max_sw=official)
    AXIS_ITERS = gap_end
    print(f"[figures] fig3-5 axis 0-{AXIS_ITERS}  (markers every {_mark_period()})")
    fig_gap(traj, figdir)
    fig_bs_bids(traj, figdir)
    fig_hsp_bids(traj, figdir)
    fig_omega_const(mkt["omega"], figdir, mkt["hsp_names"])
    persist_arrays(traj, figdir, mkt["hsp_names"])
    if copy_to_report:
        dest = ROOT / "report" / "current"
        dest.mkdir(parents=True, exist_ok=True)
        for p in figdir.glob("fig*.png"):
            (dest / p.name).write_bytes(p.read_bytes())
    print(f"[figures] wrote {figdir}")
    return figdir


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate paper-style auction result figures.")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR)
    p.add_argument("--max-iter", type=int, default=450)
    p.add_argument("--omega-path", type=Path, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    generate_figures(args.processed_dir, args.figdir, args.max_iter, omega_path=args.omega_path)
