#!/usr/bin/env python
"""Plot the three figures used in the Visualization Analysis section.

The script is intentionally data-source tolerant:

1. Training reward curves can be reconstructed from DEBUG accuracy logs
   (`debug_logs/accuracy_*.log`) or read from W&B/CSV exports.
2. Learnability dynamics and landscape are reconstructed from sampled accuracy
   rewards, grouped by `num_generations`.

Expected outputs:
    figures/training_curves.pdf
    figures/learnability_dynamics.pdf
    figures/learnability_landscape.pdf
    figures/visualization_group_stats.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_RS_DIR = Path(
    "Qwen2.5-VL-7B-Instruct_clevrer_acclearn_twgrpo_lam015_tau05"
)


@dataclass(frozen=True)
class Style:
    tw_color: str = "#2F6B9A"
    rs_color: str = "#C43C39"
    accent: str = "#D9902F"
    dark: str = "#222222"
    grid: str = "#D9D9D9"


STYLE = Style()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rolling_mean(values: Iterable[float], window: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or window <= 1:
        return arr
    series = pd.Series(arr)
    return series.rolling(window=window, min_periods=max(1, window // 5)).mean().to_numpy()


def read_accuracy_rewards_from_logs(run_dir: Path) -> pd.DataFrame:
    """Read per-sample accuracy rewards from debug_logs/accuracy_*.log."""
    log_dir = run_dir / "debug_logs"
    log_paths = sorted(log_dir.glob("accuracy_*.log"))
    if not log_paths:
        log_dir = run_dir
        log_paths = sorted(log_dir.glob("accuracy_*.log"))
    rows: list[dict[str, float | int | str]] = []
    reward_re = re.compile(r"Calculated reward:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")

    for process_idx, log_path in enumerate(log_paths):
        local_index = 0
        with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                match = reward_re.search(line)
                if not match:
                    continue
                rows.append(
                    {
                        "process": process_idx,
                        "log_file": log_path.name,
                        "sample_index": local_index,
                        "reward": float(match.group(1)),
                    }
                )
                local_index += 1

    if not rows:
        raise FileNotFoundError(
            f"No accuracy rewards found under {log_dir}. "
            "Expected accuracy_*.log in the run directory or debug_logs/. "
            "Enable DEBUG_MODE=true during training or pass CSV metrics."
        )

    return pd.DataFrame(rows)


def group_accuracy_rewards(
    rewards_df: pd.DataFrame,
    num_generations: int,
    lambda_: float,
    tau: float,
) -> pd.DataFrame:
    """Reconstruct prompt-group statistics from sequential sampled rewards."""
    if num_generations <= 0:
        raise ValueError("num_generations must be positive.")

    grouped_rows: list[dict[str, float | int | str]] = []
    for (process, log_file), proc_df in rewards_df.groupby(["process", "log_file"], sort=True):
        proc_df = proc_df.sort_values("sample_index")
        rewards = proc_df["reward"].to_numpy(dtype=float)
        usable = (len(rewards) // num_generations) * num_generations
        if usable != len(rewards):
            warnings.warn(
                f"{log_file}: dropping {len(rewards) - usable} trailing rewards "
                f"because they do not form a full group of {num_generations}."
            )
        rewards = rewards[:usable].reshape(-1, num_generations)
        for group_idx, group_rewards in enumerate(rewards, start=1):
            max_reward = float(np.max(group_rewards))
            min_reward = float(np.min(group_rewards))
            mean_reward = float(np.mean(group_rewards))
            contrast = max_reward - min_reward
            unsolved = float(np.clip(1.0 - mean_reward, 0.0, 1.0))
            best_quality = 0.5 + 0.5 * max_reward
            learnability_raw = contrast * unsolved * best_quality
            learnability = float(np.clip(learnability_raw / tau, 0.0, 1.0))
            omega = float(1.0 + lambda_ * (2.0 * learnability - 1.0))
            grouped_rows.append(
                {
                    "process": int(process),
                    "log_file": str(log_file),
                    "step": group_idx,
                    "mean_accuracy_reward": mean_reward,
                    "min_accuracy_reward": min_reward,
                    "max_accuracy_reward": max_reward,
                    "reward_contrast": contrast,
                    "unsolvedness": unsolved,
                    "best_response_quality": best_quality,
                    "learnability": learnability,
                    "omega": omega,
                    "amplified": float(omega > 1.0),
                }
            )

    if not grouped_rows:
        raise ValueError("No complete prompt groups could be reconstructed.")
    return pd.DataFrame(grouped_rows)


def aggregate_group_stats_by_step(group_stats: pd.DataFrame) -> pd.DataFrame:
    return (
        group_stats.groupby("step", as_index=False)
        .agg(
            train_accuracy_reward=("mean_accuracy_reward", "mean"),
            learnability=("learnability", "mean"),
            omega=("omega", "mean"),
            amplified_ratio=("amplified", "mean"),
            reward_contrast=("reward_contrast", "mean"),
        )
        .sort_values("step")
    )


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower_to_original = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    for c in columns:
        c_lower = c.lower()
        if any(cand.lower() in c_lower for cand in candidates):
            return c
    return None


def load_metric_csv(
    path: Path,
    metric_candidates: Iterable[str],
    value_name: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    step_col = find_column(df.columns, ["step", "global_step", "_step", "Step", "trainer/global_step"])
    metric_col = find_column(df.columns, metric_candidates)
    if step_col is None or metric_col is None:
        raise ValueError(
            f"Could not identify step/metric columns in {path}. "
            f"Available columns: {list(df.columns)}"
        )
    out = df[[step_col, metric_col]].rename(columns={step_col: "step", metric_col: value_name})
    out = out.dropna()
    out["step"] = out["step"].astype(float)
    out[value_name] = out[value_name].astype(float)
    return out.sort_values("step")


def load_train_curve(
    run_dir: Path | None,
    csv_path: Path | None,
    group_stats: pd.DataFrame | None,
    label: str,
) -> pd.DataFrame:
    if csv_path:
        return load_metric_csv(
            csv_path,
            [
                "rewards/accuracy_reward",
                "train/rewards/accuracy_reward",
                "accuracy_reward",
                "train_accuracy_reward",
            ],
            "train_accuracy_reward",
        )
    if group_stats is not None:
        return aggregate_group_stats_by_step(group_stats)[["step", "train_accuracy_reward"]]
    if run_dir is not None:
        rewards = read_accuracy_rewards_from_logs(run_dir)
        stats = group_accuracy_rewards(rewards, num_generations=8, lambda_=0.15, tau=0.5)
        return aggregate_group_stats_by_step(stats)[["step", "train_accuracy_reward"]]
    raise FileNotFoundError(f"No train reward source is available for {label}.")


def plot_training_curves(
    out_dir: Path,
    rs_train: pd.DataFrame,
    tw_train: pd.DataFrame | None,
    smooth_window: int,
) -> None:
    fig, ax1 = plt.subplots(figsize=(3.55, 2.45))

    def plot_train(df: pd.DataFrame, label: str, color: str) -> None:
        y = rolling_mean(df["train_accuracy_reward"], smooth_window)
        ax1.plot(df["step"], y, color=color, lw=1.9, label=f"{label} train reward")

    if tw_train is not None and not tw_train.empty:
        plot_train(tw_train, "TW-GRPO", STYLE.tw_color)
    plot_train(rs_train, "RS-GRPO", STYLE.rs_color)

    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Training accuracy reward")
    ax1.set_ylim(0.0, min(1.02, max(1.0, ax1.get_ylim()[1])))
    ax1.grid(True, color=STYLE.grid, lw=0.5, alpha=0.7)
    ax1.set_axisbelow(True)

    ax1.legend(loc="lower right", frameon=False, handlelength=2.4)
    fig.savefig(out_dir / "training_curves.pdf")
    fig.savefig(out_dir / "training_curves.png", dpi=300)
    plt.close(fig)


def plot_learnability_dynamics(
    out_dir: Path,
    step_stats: pd.DataFrame,
    smooth_window: int,
) -> None:
    fig, ax1 = plt.subplots(figsize=(3.55, 2.35))
    ax2 = ax1.twinx()

    steps = step_stats["step"]
    learnability = rolling_mean(step_stats["learnability"], smooth_window)
    amplified = rolling_mean(step_stats["amplified_ratio"], smooth_window)

    line1 = ax1.plot(
        steps,
        learnability,
        color=STYLE.rs_color,
        lw=1.9,
        label=r"Mean learnability $L(q)$",
    )
    line2 = ax2.plot(
        steps,
        amplified,
        color=STYLE.tw_color,
        lw=1.7,
        ls="--",
        label=r"Amplified groups $(\Omega(q)>1)$",
    )

    ax1.set_xlabel("Training step")
    ax1.set_ylabel(r"Mean $L(q)$")
    ax2.set_ylabel("Amplified group ratio")
    ax1.set_ylim(0.0, min(1.0, max(0.25, float(np.nanmax(learnability)) * 1.15)))
    ax2.set_ylim(0.0, 1.0)
    ax1.grid(True, color=STYLE.grid, lw=0.5, alpha=0.7)
    ax1.set_axisbelow(True)

    lines = line1 + line2
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=False)
    fig.savefig(out_dir / "learnability_dynamics.pdf")
    fig.savefig(out_dir / "learnability_dynamics.png", dpi=300)
    plt.close(fig)


def plot_learnability_landscape(
    out_dir: Path,
    group_stats: pd.DataFrame,
    lambda_: float,
    tau: float,
    max_points: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    plot_df = group_stats.copy()
    if len(plot_df) > max_points:
        plot_df = plot_df.iloc[rng.choice(len(plot_df), size=max_points, replace=False)].copy()

    jitter_scale = 0.008
    x = np.clip(
        plot_df["reward_contrast"].to_numpy(dtype=float)
        + rng.normal(0.0, jitter_scale, len(plot_df)),
        0.0,
        1.0,
    )
    y = np.clip(
        plot_df["mean_accuracy_reward"].to_numpy(dtype=float)
        + rng.normal(0.0, jitter_scale, len(plot_df)),
        0.0,
        1.0,
    )

    fig, ax = plt.subplots(figsize=(3.55, 2.75))
    scatter = ax.scatter(
        x,
        y,
        c=plot_df["omega"],
        cmap="viridis",
        vmin=1.0 - lambda_,
        vmax=1.0 + lambda_,
        s=11,
        alpha=0.66,
        linewidths=0,
    )

    ax.set_xlabel(r"Reward contrast $C(q)$")
    ax.set_ylabel(r"Mean accuracy reward $\bar{r}$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, color=STYLE.grid, lw=0.5, alpha=0.55)
    ax.set_axisbelow(True)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02, fraction=0.055)
    cbar.set_label(r"Group weight $\Omega(q)$")
    fig.savefig(out_dir / "learnability_landscape.pdf")
    fig.savefig(out_dir / "learnability_landscape.png", dpi=300)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rs-dir", type=Path, default=DEFAULT_RS_DIR)
    parser.add_argument("--tw-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--lambda-weight", type=float, default=0.15)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--smooth-window", type=int, default=35)
    parser.add_argument("--max-landscape-points", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rs-train-csv", type=Path, default=None)
    parser.add_argument("--tw-train-csv", type=Path, default=None)
    parser.add_argument(
        "--skip-training-curves",
        action="store_true",
        help="Only plot learnability dynamics and landscape.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_matplotlib()
    ensure_out_dir(args.out_dir)

    rewards_df = read_accuracy_rewards_from_logs(args.rs_dir)
    group_stats = group_accuracy_rewards(
        rewards_df,
        num_generations=args.num_generations,
        lambda_=args.lambda_weight,
        tau=args.tau,
    )
    step_stats = aggregate_group_stats_by_step(group_stats)
    group_stats.to_csv(args.out_dir / "visualization_group_stats.csv", index=False)

    if not args.skip_training_curves:
        rs_train = load_train_curve(args.rs_dir, args.rs_train_csv, group_stats, "RS-GRPO")
        tw_train = None
        if args.tw_train_csv or args.tw_dir:
            try:
                tw_train = load_train_curve(args.tw_dir, args.tw_train_csv, None, "TW-GRPO")
            except Exception as exc:
                warnings.warn(f"TW-GRPO training curve is unavailable: {exc}")
        else:
            warnings.warn("TW-GRPO training curve is unavailable; pass --tw-dir or --tw-train-csv.")

        plot_training_curves(args.out_dir, rs_train, tw_train, args.smooth_window)

    plot_learnability_dynamics(args.out_dir, step_stats, args.smooth_window)
    plot_learnability_landscape(
        args.out_dir,
        group_stats,
        lambda_=args.lambda_weight,
        tau=args.tau,
        max_points=args.max_landscape_points,
        seed=args.seed,
    )

    print(f"Wrote figures and group stats to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
