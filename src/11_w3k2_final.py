"""
Paper-1 topology experiments: 3x2, 5x3, and 10x5 (HSP x BS).

Each run writes tables and figures into its own folder so the curves stay
separate. Subgradient traces use Delta in {0.01, 0.02, 0.04, 0.08}.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import load_step

SCORE_SRC = ROOT / "data" / "processed_w2k3"
SEED = 42
AREA_M = 2000.0
FIG_DELTAS = (0.01, 0.02, 0.04, 0.08)
FIG_PI_INIT = 0.50
FIG_MAX_ITER = 700
FIG345_DELTA = 0.04
SCALE_NS = (10, 50, 100, 500, 1000, 5000, 20000)
FULL_RATIOS_3 = (
    (1, 1, 1),
    (4, 3, 3),
    (5, 4, 3),
    (5, 4, 1),
    (7, 2, 1),
    (8, 1, 1),
    (2, 1, 1),
    (3, 2, 1),
    (9, 1, 0),
    (10, 0, 0),
)
HSP_COLOR_BASE = (
    "#2ca02c",
    "#1f77b4",
    "#d62728",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


@dataclass
class Experiment:
    name: str
    n_hsp: int
    n_bs: int
    out_dir: Path
    market_ratio: tuple[int, ...]

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        if len(self.market_ratio) != self.n_hsp:
            raise ValueError(f"{self.name}: ratio length {len(self.market_ratio)} != n_hsp={self.n_hsp}")

    @property
    def market_dir(self) -> Path:
        return self.out_dir / "market"

    @property
    def sweep_dir(self) -> Path:
        return self.out_dir / "preference_sweep"

    @property
    def fig_dir(self) -> Path:
        return self.out_dir / "figures"

    @property
    def hsp_names(self) -> tuple[str, ...]:
        return tuple(f"HSP{i + 1}" for i in range(self.n_hsp))

    @property
    def bs_names(self) -> tuple[str, ...]:
        return tuple(f"BS{i + 1}" for i in range(self.n_bs))

    @property
    def hsp_colors(self) -> list[str]:
        if self.n_hsp <= len(HSP_COLOR_BASE):
            return list(HSP_COLOR_BASE[: self.n_hsp])
        cmap = plt.get_cmap("tab20")
        return [cmap(i % 20) for i in range(self.n_hsp)]


def topologies() -> list[Experiment]:
    return [
        Experiment("3x2", 3, 2, ROOT / "3x2 experiment", (5, 4, 3)),
        Experiment("5x3", 5, 3, ROOT / "5x3 experiment", (5, 4, 3, 2, 1)),
        Experiment("10x5", 10, 5, ROOT / "10x5 experiment", tuple(range(10, 0, -1))),
    ]


def migrate_legacy_3x2() -> None:
    old = ROOT / "final result"
    new = ROOT / "3x2 experiment"
    if old.exists() and not new.exists():
        old.rename(new)
        print(f"[final] renamed {old.name} -> {new.name}")


def extra_ratios(n_hsp: int) -> tuple[tuple[int, ...], ...]:
    if n_hsp == 3:
        return FULL_RATIOS_3
    equal = tuple([1] * n_hsp)
    decreasing = tuple(range(n_hsp, 0, -1))
    spike = (max(3 * n_hsp, 8),) + (1,) * (n_hsp - 1)
    front = (n_hsp + 2, n_hsp, max(n_hsp - 2, 1)) + tuple(
        max(n_hsp - 3 - i, 1) for i in range(max(n_hsp - 3, 0))
    )
    front = front[:n_hsp]
    return (equal, decreasing, front, spike)


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
        }
    )


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def compositions(n: int, parts: int = 3) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(n,)]
    out: list[tuple[int, ...]] = []
    for first in range(n + 1):
        for rest in compositions(n - first, parts - 1):
            out.append((first, *rest))
    return out


def counts_from_ratio(n: int, ratio: tuple[int, ...]) -> tuple[int, ...]:
    weights = np.asarray(ratio, dtype=np.float64)
    if float(weights.sum()) <= 0:
        out = [0] * len(ratio)
        out[0] = n
        return tuple(out)
    raw = n * weights / weights.sum()
    counts = np.floor(raw).astype(int)
    leftover = n - int(counts.sum())
    order = np.argsort(-(raw - counts))
    for i in range(leftover):
        counts[order[i % len(counts)]] += 1
    return tuple(int(x) for x in counts)


def assign_counts(n: int, counts: tuple[int, ...], seed: int) -> np.ndarray:
    if sum(counts) != n:
        raise ValueError(f"counts {counts} do not sum to n={n}")
    labels = np.concatenate([np.full(c, i, dtype=np.int64) for i, c in enumerate(counts)])
    rng = np.random.default_rng(seed)
    rng.shuffle(labels)
    return labels


def rho_for_subset(agg, exp: Experiment, crit, xy, bs_xy, index, hsp_id) -> dict:
    c = crit[index]
    pts = xy[index]
    bs_id, _, _ = agg.associate_bs(pts, bs_xy)
    c_wk, n_wk, mean_wk = agg.aggregate_links(bs_id, hsp_id, c, exp.n_bs, exp.n_hsp)
    prefs = agg.build_rho(c_wk, n_wk, mean_wk)
    rho = prefs["rho_wk"]
    c_bar = float(np.mean(c_wk) + agg.EPS)
    raw = (c_wk / c_bar) * (1.0 + mean_wk)
    raw = np.where(n_wk > 0, raw, 0.0)
    return {
        "rho": rho,
        "raw": raw,
        "C_wk": c_wk,
        "N_wk": n_wk,
        "mean_c": mean_wk,
        "bs_id": bs_id,
        "hsp_id": hsp_id,
        "mean_rho_hsp": rho.mean(axis=0),
        "mean_raw_hsp": raw.mean(axis=0),
        "mass_hsp": c_wk.sum(axis=0),
    }


def prepare_market_dir(exp: Experiment) -> None:
    exp.market_dir.mkdir(parents=True, exist_ok=True)
    scores = SCORE_SRC / "criticality_scores.csv"
    if not scores.exists():
        raise FileNotFoundError(f"Need {scores} (run fusion on eICU first).")
    dest = exp.market_dir / "criticality_scores.csv"
    if not dest.exists() or dest.stat().st_size != scores.stat().st_size:
        shutil.copy2(scores, dest)
    meta = SCORE_SRC / "patients_all.csv"
    if meta.exists():
        shutil.copy2(meta, exp.market_dir / "patients_all.csv")


def run_clearing(exp: Experiment) -> Path:
    prepare_market_dir(exp)
    agg = load_step("aggregator")
    rel = load_step("reluctance")
    opt = load_step("optimizer")
    agg.run_aggregator(
        agg.AggregatorConfig(
            processed_dir=exp.market_dir,
            artifact_dir=exp.market_dir,
            n_bs=exp.n_bs,
            n_hsp=exp.n_hsp,
            area_m=AREA_M,
            seed=SEED,
            hsp_ratio=exp.market_ratio,
        )
    )
    omega_path = rel.write_frozen_omega(exp.market_dir, exp.market_dir / "reluctance", seed=SEED)
    opt.run_optimizer(
        opt.OptimizerConfig(
            processed_dir=exp.market_dir,
            artifact_dir=exp.market_dir,
            clearing_dir=exp.market_dir,
            omega_path=omega_path,
            bs_capacity_mbps=5.0,
            scale_capacity_by_load=False,
            max_iter=1500,
            step0=0.08,
            pi_init=0.32,
        )
    )
    return omega_path


def run_market_figures(exp: Experiment, omega_path: Path) -> None:
    figs = load_step("auction_figures")
    econ = load_step("economic_figures")
    exp.fig_dir.mkdir(parents=True, exist_ok=True)
    figs.generate_figures(
        processed_dir=exp.market_dir,
        figdir=exp.fig_dir,
        max_iter=FIG_MAX_ITER,
        omega_path=omega_path,
        copy_to_report=False,
        deltas=FIG_DELTAS,
        pi_init=FIG_PI_INIT,
        fig345_delta=FIG345_DELTA,
    )
    econ_iters = 80 if exp.n_hsp >= 8 else 150
    econ.generate_economic_figures(
        processed_dir=exp.market_dir,
        figdir=exp.fig_dir,
        max_iter=econ_iters,
        clearing_dir=exp.market_dir,
        omega_path=omega_path,
        copy_to_report=False,
    )


def load_pool(agg, exp: Experiment) -> dict:
    cfg = agg.AggregatorConfig(
        processed_dir=exp.market_dir,
        n_bs=exp.n_bs,
        n_hsp=exp.n_hsp,
        seed=SEED,
    )
    customers = agg._load_customers(cfg)
    n = len(customers)
    crit = customers["criticality"].to_numpy(dtype=np.float64)
    xy = agg.place_customers(n, AREA_M, SEED)
    bs_xy = agg.place_base_stations(exp.n_bs, AREA_M)
    bs_id, dist_m, rsrp = agg.associate_bs(xy, bs_xy)
    hsp_id = agg.assign_hsp(n, exp.n_hsp, SEED, ratio=exp.market_ratio)
    return {
        "customers": customers,
        "crit": crit,
        "xy": xy,
        "bs_xy": bs_xy,
        "bs_id": bs_id,
        "hsp_id": hsp_id,
        "dist_m": dist_m,
        "rsrp": rsrp,
    }


def _record_pack(exp: Experiment, n_total: int, counts: tuple[int, ...], pack: dict, ratio=None) -> dict:
    rec = {
        "n_total": n_total,
        "split": "-".join(str(c) for c in counts),
        "ratio": ":".join(str(x) for x in ratio) if ratio is not None else "",
        "imbalance": max(counts) / n_total if n_total else 0.0,
    }
    for j, hsp in enumerate(exp.hsp_names):
        rec[f"n_{hsp.lower()}"] = int(counts[j])
        rec[f"share_{hsp.lower()}"] = counts[j] / n_total if n_total else 0.0
        rec[f"mean_rho_{hsp}"] = float(pack["mean_rho_hsp"][j])
        rec[f"mean_raw_{hsp}"] = float(pack["mean_raw_hsp"][j])
        rec[f"mass_{hsp}"] = float(pack["mass_hsp"][j])
    for i, bs in enumerate(exp.bs_names):
        for j, hsp in enumerate(exp.hsp_names):
            rec[f"rho_{bs}_{hsp}"] = float(pack["rho"][i, j])
            rec[f"N_{bs}_{hsp}"] = int(pack["N_wk"][i, j])
    return rec


def run_preference_sweep(agg, exp: Experiment, pool: dict) -> dict:
    exp.sweep_dir.mkdir(parents=True, exist_ok=True)
    n_all = len(pool["crit"])
    rng = np.random.default_rng(SEED + 3)
    order = rng.permutation(n_all)
    full_index = np.arange(n_all)
    official = tuple(exp.market_ratio)

    full_rows = []
    for ratio in extra_ratios(exp.n_hsp):
        counts = counts_from_ratio(n_all, ratio)
        if tuple(ratio) == official:
            hsp_id = pool["hsp_id"]
        else:
            hsp_id = assign_counts(n_all, counts, seed=SEED + 21 * sum(ratio) + ratio[0])
        pack = rho_for_subset(agg, exp, pool["crit"], pool["xy"], pool["bs_xy"], full_index, hsp_id)
        full_rows.append(_record_pack(exp, n_all, counts, pack, ratio=ratio))
    full_df = pd.DataFrame(full_rows)
    full_df.to_csv(exp.sweep_dir / "preference_vs_split.csv", index=False)

    share_rows = []
    for share in np.round(np.linspace(0.05, 0.90, 18), 4):
        n1 = int(round(share * n_all))
        rem = n_all - n1
        base, leftover = divmod(rem, max(exp.n_hsp - 1, 1))
        rest = [base] * (exp.n_hsp - 1)
        for i in range(leftover):
            rest[i % len(rest)] += 1
        counts = tuple([n1] + rest)
        hsp_id = assign_counts(n_all, counts, seed=SEED + 33)
        pack = rho_for_subset(agg, exp, pool["crit"], pool["xy"], pool["bs_xy"], full_index, hsp_id)
        share_rows.append(_record_pack(exp, n_all, counts, pack))
    share_df = pd.DataFrame(share_rows)
    share_df.to_csv(exp.sweep_dir / "preference_vs_share.csv", index=False)

    toy_df = pd.DataFrame()
    if exp.n_hsp == 3:
        toy_rows = []
        for counts in compositions(10) + [(5, 4, 3)]:
            n_total = sum(counts)
            index = order[:n_total]
            hsp_id = assign_counts(n_total, counts, seed=SEED + 1000 * n_total + counts[0])
            pack = rho_for_subset(agg, exp, pool["crit"], pool["xy"], pool["bs_xy"], index, hsp_id)
            toy_rows.append(_record_pack(exp, n_total, counts, pack))
        toy_df = pd.DataFrame(toy_rows)
        toy_df.to_csv(exp.sweep_dir / "preference_vs_split_N10_toy.csv", index=False)

    scale_rows = []
    for n_total in SCALE_NS + (n_all,):
        index = order[:n_total]
        base, rem = divmod(n_total, exp.n_hsp)
        counts = tuple(base + (1 if j < rem else 0) for j in range(exp.n_hsp))
        hsp_id = assign_counts(n_total, counts, seed=SEED + 9)
        pack = rho_for_subset(agg, exp, pool["crit"], pool["xy"], pool["bs_xy"], index, hsp_id)
        scale_rows.append(_record_pack(exp, n_total, counts, pack))
    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(exp.sweep_dir / "preference_vs_n_balanced.csv", index=False)

    _ieee_rc()
    _fig_full_ratios(exp, full_df)
    _fig_share_sweep(exp, share_df)
    _fig_scale(exp, scale_df)
    if exp.n_hsp == 3 and not toy_df.empty:
        _fig_highlight_splits(exp, toy_df)
        _fig_composition_scatter(exp, toy_df)
        _fig_raw_vs_count(exp, toy_df)
    return {"sweep": full_df, "share": share_df, "scale": scale_df, "toy": toy_df}


def _fig_full_ratios(exp: Experiment, df: pd.DataFrame) -> None:
    colors = exp.hsp_colors
    if exp.n_hsp <= 5:
        labels = []
        for row in df.itertuples():
            counts = [int(getattr(row, f"n_{h.lower()}")) for h in exp.hsp_names]
            labels.append(f"{row.ratio}\n(" + "-".join(f"{c // 1000}k" for c in counts) + ")")
        x = np.arange(len(df))
        width = 0.8 / max(exp.n_hsp, 1)
        fig, ax = plt.subplots(figsize=(max(8.2, 1.15 * len(df) + 2.0), 3.8))
        for j, hsp in enumerate(exp.hsp_names):
            ax.bar(
                x + (j - (exp.n_hsp - 1) / 2) * width,
                df[f"mean_rho_{hsp}"].to_numpy(),
                width,
                label=hsp,
                color=colors[j],
                edgecolor="k",
                linewidth=0.3,
            )
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylabel(r"Mean $\rho$ of that HSP")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(f"Share of all users (example ratios, {exp.n_hsp} HSPs)")
        ax.legend(framealpha=0.92, ncol=min(exp.n_hsp, 5), fontsize=7)
        fig.tight_layout()
    else:
        mat = np.vstack([df[f"mean_rho_{hsp}"].to_numpy() for hsp in exp.hsp_names])
        fig, ax = plt.subplots(figsize=(max(6.8, 0.9 * len(df) + 2.2), 4.4))
        im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_yticks(np.arange(exp.n_hsp), exp.hsp_names)
        ax.set_xticks(np.arange(len(df)), [str(r) for r in df["ratio"]], rotation=30, ha="right")
        ax.set_xlabel("Share ratio")
        fig.colorbar(im, ax=ax, fraction=0.046, label=r"Mean $\rho$")
        fig.tight_layout()
    _save(fig, exp.sweep_dir / "fig_preference_selected_splits.png")


def _fig_share_sweep(exp: Experiment, df: pd.DataFrame) -> None:
    colors = exp.hsp_colors
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    share_col = f"share_{exp.hsp_names[0].lower()}"
    ax = axes[0]
    for j, hsp in enumerate(exp.hsp_names):
        ax.plot(df[share_col], df[f"mass_{hsp}"], color=colors[j], linewidth=1.4, label=hsp if exp.n_hsp <= 5 else None)
    ax.set_xlabel(f"Share of all users given to {exp.hsp_names[0]}")
    ax.set_ylabel("Criticality mass C of that HSP")
    if exp.n_hsp <= 5:
        ax.legend(framealpha=0.92, fontsize=7, ncol=1)
    ax = axes[1]
    for j, hsp in enumerate(exp.hsp_names):
        ax.plot(df[share_col], df[f"mean_rho_{hsp}"], color=colors[j], linewidth=1.4, label=hsp if exp.n_hsp <= 5 else None)
    ax.set_xlabel(f"Share of all users given to {exp.hsp_names[0]}")
    ax.set_ylabel(r"Mean $\rho$ (max-normalized)")
    ax.set_ylim(0.0, 1.05)
    if exp.n_hsp <= 5:
        ax.legend(framealpha=0.92, fontsize=7)
        fig.tight_layout()
    else:
        for ax_i in axes:
            for line, j in zip(ax_i.lines, range(exp.n_hsp)):
                line.set_color(plt.cm.viridis(j / max(exp.n_hsp - 1, 1)))
        fig.subplots_adjust(right=0.90, wspace=0.28)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(1, exp.n_hsp))
        sm.set_array([])
        fig.colorbar(sm, ax=list(axes), fraction=0.04, pad=0.03, label="HSP index")
    _save(fig, exp.sweep_dir / "fig_preference_vs_share.png")


def _fig_highlight_splits(exp: Experiment, df: pd.DataFrame) -> None:
    wanted = {"4-3-3", "5-4-1", "7-2-1", "8-1-1", "10-0-0", "5-4-3"}
    sub = df[df["split"].isin(wanted)].drop_duplicates("split")
    order = [s for s in ("4-3-3", "5-4-1", "7-2-1", "8-1-1", "10-0-0", "5-4-3") if s in set(sub["split"])]
    sub = sub.set_index("split").loc[order]
    links = [f"rho_{bs}_{hsp}" for bs in exp.bs_names for hsp in exp.hsp_names]
    x = np.arange(len(order))
    width = 0.8 / max(len(links), 1)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for i, link in enumerate(links):
        ax.bar(
            x + (i - (len(links) - 1) / 2) * width,
            sub[link].to_numpy(),
            width,
            label=link.replace("rho_", "").replace("_", "–"),
            edgecolor="k",
            linewidth=0.3,
        )
    ax.set_xticks(x, order)
    ax.set_xlabel("Toy split of 10 patients (not the real market)")
    ax.set_ylabel(r"Preference $\rho_{wk}$ (max-normalized)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(ncol=3, fontsize=7, framealpha=0.92)
    fig.tight_layout()
    _save(fig, exp.sweep_dir / "fig_preference_selected_splits_N10_toy.png")


def _fig_composition_scatter(exp: Experiment, df: pd.DataFrame) -> None:
    ten = df[df["n_total"] == 10]
    colors = exp.hsp_colors
    fig, axes = plt.subplots(1, exp.n_hsp, figsize=(8.8, 3.3), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for j, hsp in enumerate(exp.hsp_names):
        ax = axes[j]
        ax.scatter(
            ten[f"n_{hsp.lower()}"],
            ten[f"mean_rho_{hsp}"],
            s=18,
            c=colors[j],
            edgecolors="k",
            linewidths=0.25,
            alpha=0.75,
        )
        ax.set_xlabel(f"Patients on {hsp} (out of 10)")
        ax.set_title(hsp)
    axes[0].set_ylabel(r"Mean $\rho$ of that HSP")
    fig.tight_layout()
    _save(fig, exp.sweep_dir / "fig_preference_vs_count_N10.png")

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for j, hsp in enumerate(exp.hsp_names):
        ax.scatter(ten["imbalance"], ten[f"mean_rho_{hsp}"], s=16, c=colors[j], label=hsp, alpha=0.75)
    ax.set_xlabel(r"Imbalance $\max_w n_w / 10$")
    ax.set_ylabel(r"Mean $\rho$ of each HSP")
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    _save(fig, exp.sweep_dir / "fig_preference_vs_imbalance_N10.png")


def _fig_raw_vs_count(exp: Experiment, df: pd.DataFrame) -> None:
    ten = df[df["n_total"] == 10]
    colors = exp.hsp_colors
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    ax = axes[0]
    for j, hsp in enumerate(exp.hsp_names):
        ax.scatter(
            ten[f"n_{hsp.lower()}"],
            ten[f"mass_{hsp}"],
            s=18,
            c=colors[j],
            edgecolors="k",
            linewidths=0.25,
            alpha=0.75,
            label=hsp,
        )
    ax.set_xlabel("Patients on that HSP (out of 10)")
    ax.set_ylabel("Criticality mass C of that HSP")
    ax.legend(framealpha=0.92)
    ax = axes[1]
    for j, hsp in enumerate(exp.hsp_names):
        ax.scatter(
            ten[f"n_{hsp.lower()}"],
            ten[f"mean_raw_{hsp}"],
            s=18,
            c=colors[j],
            edgecolors="k",
            linewidths=0.25,
            alpha=0.75,
            label=hsp,
        )
    ax.set_xlabel("Patients on that HSP (out of 10)")
    ax.set_ylabel("Raw preference (before max-normalize)")
    ax.legend(framealpha=0.92)
    fig.tight_layout()
    _save(fig, exp.sweep_dir / "fig_raw_preference_vs_count_N10.png")


def _fig_scale(exp: Experiment, df: pd.DataFrame) -> None:
    colors = exp.hsp_colors
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    if exp.n_bs * exp.n_hsp <= 8:
        markers = ("o", "s", "^", "v", "D")
        for i, bs in enumerate(exp.bs_names):
            for j, hsp in enumerate(exp.hsp_names):
                ax.plot(
                    df["n_total"],
                    df[f"rho_{bs}_{hsp}"],
                    marker=markers[i % len(markers)],
                    color=colors[j],
                    linestyle=("-" if i == 0 else "--"),
                    label=f"{bs}–{hsp}",
                )
        ax.legend(ncol=2, fontsize=7, framealpha=0.92)
    else:
        for j, hsp in enumerate(exp.hsp_names):
            ax.plot(
                df["n_total"],
                df[f"mean_rho_{hsp}"],
                color=plt.cm.viridis(j / max(exp.n_hsp - 1, 1)),
                linewidth=1.5,
                label=hsp if exp.n_hsp <= 5 else None,
            )
        if exp.n_hsp <= 5:
            ax.legend(ncol=2, fontsize=7, framealpha=0.92)
        else:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(1, exp.n_hsp))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.046, label="HSP index")
    ax.set_xscale("log")
    ax.set_xlabel(f"Number of patients (balanced {exp.n_hsp}-way split)")
    ax.set_ylabel(r"Preference $\rho$")
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    _save(fig, exp.sweep_dir / "fig_preference_vs_n_balanced.png")


def fig_allocation_map(exp: Experiment, pool: dict) -> None:
    _ieee_rc()
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    rng = np.random.default_rng(SEED)
    take = rng.choice(len(pool["xy"]), size=min(1600, len(pool["xy"])), replace=False)
    colors = exp.hsp_colors
    for j, _hsp in enumerate(exp.hsp_names):
        mask = pool["hsp_id"][take] == j
        ax.scatter(
            pool["xy"][take][mask, 0],
            pool["xy"][take][mask, 1],
            s=3,
            c=colors[j],
            alpha=0.28,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
    bs_xy = np.asarray(pool["bs_xy"], dtype=np.float64)
    if len(bs_xy) == 2:
        mid = 0.5 * (bs_xy[0] + bs_xy[1])
        delta = bs_xy[1] - bs_xy[0]
        nrm = np.array([-delta[1], delta[0]], dtype=np.float64)
        nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-9)
        span = 1.2 * AREA_M
        ax.plot(
            [mid[0] - span * nrm[0], mid[0] + span * nrm[0]],
            [mid[1] - span * nrm[1], mid[1] + span * nrm[1]],
            color="0.35",
            linestyle=":",
            linewidth=1.0,
            zorder=2,
        )
    ax.scatter(
        bs_xy[:, 0],
        bs_xy[:, 1],
        marker="^",
        s=55,
        c="k",
        edgecolors="w",
        linewidths=0.7,
        zorder=5,
    )
    for i, name in enumerate(exp.bs_names):
        ax.annotate(
            name,
            bs_xy[i],
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=8,
            fontweight="bold",
            zorder=6,
        )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[j],
            markeredgecolor="none",
            markersize=6,
            label=exp.hsp_names[j],
        )
        for j in range(exp.n_hsp)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="k",
            markeredgecolor="k",
            markersize=8,
            label="BS",
        )
    )
    ncol = 4 if exp.n_hsp <= 3 else (5 if exp.n_hsp <= 9 else 6)
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14 if exp.n_hsp > 5 else 1.12),
        ncol=ncol,
        framealpha=0.95,
        fontsize=7 if exp.n_hsp > 5 else 8,
        handletextpad=0.35,
        columnspacing=0.9,
        borderpad=0.35,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(0, AREA_M)
    ax.set_ylim(0, AREA_M)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.yaxis.set_major_locator(MultipleLocator(500))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90 if exp.n_hsp > 5 else 0.93))
    _save(fig, exp.sweep_dir / "fig_allocation_map.png")


def write_allocation_tables(exp: Experiment, pool: dict, agg) -> dict:
    c_wk, n_wk, mean_wk = agg.aggregate_links(
        pool["bs_id"], pool["hsp_id"], pool["crit"], exp.n_bs, exp.n_hsp
    )
    users_hsp = n_wk.sum(axis=0)
    load_bs = n_wk.sum(axis=1)
    ratio_txt = ":".join(str(x) for x in exp.market_ratio)
    alloc = {
        "n_users": int(len(pool["crit"])),
        "n_unique_stays": int(pool["customers"]["patient_id"].nunique())
        if "patient_id" in pool["customers"].columns
        else None,
        "topology": f"{exp.n_hsp} HSP x {exp.n_bs} BS",
        "how": (
            "Each eICU vital window is one uplink user. Users are placed uniformly "
            f"on a 2 km × 2 km map (seed 42), associated to the strongest of {exp.n_bs} "
            f"cells by max RSRP, and subscribed to {', '.join(exp.hsp_names)} in a "
            f"{ratio_txt} share (seed 42+7) so hospitals sit in different preference "
            "bands. No disease-to-hospital map."
        ),
        "users_per_hsp": {exp.hsp_names[j]: int(users_hsp[j]) for j in range(exp.n_hsp)},
        "users_per_bs": {exp.bs_names[i]: int(load_bs[i]) for i in range(exp.n_bs)},
        "N_wk": {
            exp.bs_names[i]: {exp.hsp_names[j]: int(n_wk[i, j]) for j in range(exp.n_hsp)}
            for i in range(exp.n_bs)
        },
        "C_wk": {
            exp.bs_names[i]: {exp.hsp_names[j]: float(c_wk[i, j]) for j in range(exp.n_hsp)}
            for i in range(exp.n_bs)
        },
        "share_per_hsp": {
            exp.hsp_names[j]: float(users_hsp[j] / users_hsp.sum()) for j in range(exp.n_hsp)
        },
    }
    (exp.out_dir / "patient_allocation.json").write_text(json.dumps(alloc, indent=2), encoding="utf-8")
    pd.DataFrame(n_wk, index=exp.bs_names, columns=exp.hsp_names).to_csv(exp.out_dir / "N_wk.csv")
    pd.DataFrame(c_wk, index=exp.bs_names, columns=exp.hsp_names).to_csv(exp.out_dir / "C_wk.csv")
    for name in ("rho_wk.csv", "omega_wk.csv", "optimizer_summary.json"):
        src = exp.market_dir / name
        if src.exists():
            shutil.copy2(src, exp.out_dir / name)
    return alloc


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join((head, sep, body))


def write_readmes(exp: Experiment, alloc: dict, sweep: pd.DataFrame, scale: pd.DataFrame) -> None:
    opt = {}
    opt_path = exp.market_dir / "optimizer_summary.json"
    if opt_path.exists():
        opt = json.loads(opt_path.read_text(encoding="utf-8"))
    rho_path = exp.market_dir / "rho_wk.csv"
    rho_txt = rho_path.read_text(encoding="utf-8") if rho_path.exists() else ""
    omega_path = exp.market_dir / "omega_wk.csv"
    omega_txt = omega_path.read_text(encoding="utf-8") if omega_path.exists() else ""
    ratio_txt = ":".join(str(x) for x in exp.market_ratio)
    users = alloc["users_per_hsp"]
    nwk = alloc["N_wk"]
    delta_txt = ", ".join(f"{x:.2f}" for x in FIG_DELTAS)

    hsp_rows = [
        [h, str(users[h]), f"{alloc['share_per_hsp'][h]:.1%}"] for h in exp.hsp_names
    ]
    hsp_tbl = _md_table(["HSP", "Users", "Share"], hsp_rows)

    link_headers = [""] + list(exp.hsp_names) + ["BS total"]
    link_rows = []
    for bs in exp.bs_names:
        link_rows.append(
            [bs] + [str(nwk[bs][h]) for h in exp.hsp_names] + [str(alloc["users_per_bs"][bs])]
        )
    link_tbl = _md_table(link_headers, link_rows)

    examples = []
    if "ratio" in sweep.columns:
        shown = sweep["ratio"].tolist()[:5]
        for ratio in shown:
            row = sweep[sweep["ratio"] == ratio]
            if row.empty:
                continue
            r = row.iloc[0]
            tag = " (official market)" if ratio == ratio_txt else ""
            counts = "/".join(str(int(r[f"n_{h.lower()}"])) for h in exp.hsp_names)
            rhos = ", ".join(f"{h} {r[f'mean_rho_{h}']:.3f}" for h in exp.hsp_names[: min(5, exp.n_hsp)])
            examples.append(f"- **{ratio}**{tag} -> {counts} users: mean rho = {rhos}")

    (exp.out_dir / "README.md").write_text(
        f"""# {exp.name} experiment — {exp.n_hsp} HSPs × {exp.n_bs} base stations

Topology for this folder: **W = {exp.n_hsp} buyers** and **K = {exp.n_bs} sellers**.
Capacity is **5 Mbps per cell**, reluctance is the frozen one-shot radio table, preference is max-normalized.
Fig. 2 subgradient steps: **{delta_txt}**.

## How patients are allocated

{alloc["how"]}

There are **{alloc["n_users"]}** uplink users (eICU Demo vital windows). Unique ICU stays: **{alloc["n_unique_stays"]}**.

### Users per hospital ({ratio_txt} split, seed 42)

{hsp_tbl}

### Users on each (BS, HSP) link

{link_tbl}

Map (1.6k-user sample): `preference_sweep/fig_allocation_map.png`.

## Full-market preference and clearing

Official split is **{ratio_txt}**. Preference rho = (C / mean C) * (1 + mean c), then **max-normalized** so the largest link is 1.

Preference matrix (`market/rho_wk.csv`):

```
{rho_txt.strip()}
```

Frozen reluctance (`market/omega_wk.csv`):

```
{omega_txt.strip()}
```

| Metric | Value |
| --- | --- |
| Welfare | {opt.get("welfare", float("nan")):.3f} |
| Total rate (Mbps) | {opt.get("total_rate_mbps", float("nan")):.3f} |
| Total payment | {opt.get("total_payment", float("nan")):.3f} |
| Gap | {opt.get("gap", float("nan")):.3e} |
| Converged | {opt.get("converged")} |

Auction traces: `figures/fig2_convergence.png` … `fig8_preference_vs_payment.png`.

Selected full-cohort ratios:

{chr(10).join(examples)}

## Folder

```
{exp.out_dir.name}/
  README.md
  market/                 rho, omega, clearing, assignment
  figures/                auction Figs 2–8
  preference_sweep/       split experiment
```

Re-run:

```bash
python src/11_w3k2_final.py --only {exp.name}
```
""",
        encoding="utf-8",
    )
    (exp.market_dir / "README.md").write_text(
        f"W={exp.n_hsp} HSP x K={exp.n_bs} BS clearing at 5 Mbps with frozen omega.\n"
        f"Official patient split is {ratio_txt}.\n",
        encoding="utf-8",
    )
    (exp.sweep_dir / "README.md").write_text(
        f"# Preference versus patient split ({exp.name})\n\n"
        "Columns `rho_BS*_HSP*` are max-normalized preference.\n",
        encoding="utf-8",
    )


def run_experiment(exp: Experiment, skip_clear: bool, skip_figures: bool) -> None:
    exp.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[final] {exp.name}: {exp.n_hsp} HSP x {exp.n_bs} BS -> {exp.out_dir}")
    have_clear = (exp.market_dir / "optimizer_summary.json").exists()
    if skip_clear and have_clear:
        print("[final] skipping clearing (already done)")
        omega_path = exp.market_dir / "reluctance" / "omega_frozen.npz"
    else:
        omega_path = run_clearing(exp)
    if skip_figures:
        print("[final] skipping auction figures")
    else:
        run_market_figures(exp, omega_path)
    agg = load_step("aggregator")
    pool = load_pool(agg, exp)
    alloc = write_allocation_tables(exp, pool, agg)
    fig_allocation_map(exp, pool)
    sweep_pack = run_preference_sweep(agg, exp, pool)
    write_readmes(exp, alloc, sweep_pack["sweep"], sweep_pack["scale"])
    print(f"[final] {exp.name} users per HSP {alloc['users_per_hsp']}")
    print(f"[final] wrote {exp.out_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 3x2 / 5x3 / 10x5 market experiments.")
    p.add_argument("--only", choices=("3x2", "5x3", "10x5", "all"), default="all")
    p.add_argument("--sweep-only", action="store_true")
    p.add_argument("--figures-only", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    migrate_legacy_3x2()
    chosen = topologies()
    if args.only != "all":
        chosen = [e for e in chosen if e.name == args.only]
    skip_clear = args.sweep_only or args.figures_only
    skip_figures = args.sweep_only
    for exp in chosen:
        run_experiment(exp, skip_clear=skip_clear, skip_figures=skip_figures)


if __name__ == "__main__":
    main()
