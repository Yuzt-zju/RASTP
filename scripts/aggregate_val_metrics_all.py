import os
import glob
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter


BASE_DIR = "/data/zhantianyu/Project/GRID/logs/train/runs/beauty/gr"
OUTPUT_DIR = os.path.join(BASE_DIR, "aggregates")


MODES = {
    "base": "rkmeans_original_seed_*",
    "rastp_l1": "rkmeans_rastp_l1_seed_*",
    "rastp_l2": "rkmeans_rastp_l2_seed_*",
}

# 固定颜色与填充纹理，增强视觉区分度
MODE_COLORS = {
    "base": "#1f77b4",  # 蓝
    "rastp_l1": "#2ca02c",  # 绿
    "rastp_l2": "#d62728",  # 红
}

MODE_HATCHES = {
    "base": "//",
    "rastp_l1": "\\\\",
    "rastp_l2": "..",
}

# 统一调大字体，改善可读性
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


def find_metrics_files(pattern: str) -> Dict[str, str]:
    """Return mapping of seed_name -> metrics.csv path."""
    seed_dirs = glob.glob(os.path.join(BASE_DIR, pattern))
    results: Dict[str, str] = {}
    for seed_dir in seed_dirs:
        csv_path = os.path.join(seed_dir, "csv", "version_0", "metrics.csv")
        if os.path.isfile(csv_path):
            results[os.path.basename(seed_dir)] = csv_path
    return results


def load_seed_df(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # keep only step and val/* columns
    val_cols = [c for c in df.columns if c.startswith("val/")]
    keep_cols = ["step"] + val_cols
    df = df[keep_cols].copy()
    # drop rows where all val columns are NaN
    df = df.dropna(subset=val_cols, how="all")
    # sometimes step can repeat (logged twice), keep the last occurrence
    df = df.drop_duplicates(subset=["step"], keep="last").sort_values("step").reset_index(drop=True)
    return df


def intersect_common_steps(seed_to_df: Dict[str, pd.DataFrame]) -> List[int]:
    steps_sets = [set(df["step"].tolist()) for df in seed_to_df.values() if not df.empty]
    if not steps_sets:
        return []
    common = set.intersection(*steps_sets) if len(steps_sets) > 1 else steps_sets[0]
    if not common:
        return []
    return sorted(common)


def truncate_to_min_length(seed_to_df: Dict[str, pd.DataFrame]) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    # fallback when no exact step intersection: truncate by index to shortest length
    lengths = {seed: len(df) for seed, df in seed_to_df.items()}
    if not lengths:
        return [], seed_to_df
    min_len = min(lengths.values())
    seeds = sorted(seed_to_df.keys())
    return seeds, {seed: df.iloc[:min_len].reset_index(drop=True) for seed, df in seed_to_df.items()}


def aggregate_mode(mode: str, files: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    seed_to_df: Dict[str, pd.DataFrame] = {}
    for seed_name, path in files.items():
        try:
            seed_to_df[seed_name] = load_seed_df(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    if not seed_to_df:
        return pd.DataFrame(), []

    # use common steps if possible
    common_steps = intersect_common_steps(seed_to_df)
    if common_steps:
        aligned = {}
        for seed, df in seed_to_df.items():
            aligned[seed] = (
                df[df["step"].isin(common_steps)]
                .sort_values("step")
                .reset_index(drop=True)
            )
        seed_to_df = aligned
        steps = common_steps
    else:
        # fallback: truncate by index
        seeds, seed_to_df = truncate_to_min_length(seed_to_df)
        steps = seed_to_df[seeds[0]]["step"].tolist() if seeds else []

    any_df = next(iter(seed_to_df.values()))
    val_cols = [c for c in any_df.columns if c.startswith("val/")]
    # build aggregated dataframe: step + mean/std for each val col
    agg = pd.DataFrame({"step": steps})
    for col in val_cols:
        stacked = pd.concat([df[col].reset_index(drop=True) for df in seed_to_df.values()], axis=1)
        agg[f"{col}_mean"] = stacked.mean(axis=1)
        agg[f"{col}_std"] = stacked.std(axis=1, ddof=1)
    return agg, val_cols


def plot_comparison(agg_per_mode: Dict[str, pd.DataFrame], metric: str, out_path: str):
    plt.figure(figsize=(8, 5))
    for mode, agg in agg_per_mode.items():
        if agg.empty:
            continue
        x = agg["step"].values
        y = agg[f"{metric}_mean"].values
        s = agg[f"{metric}_std"].values
        color = MODE_COLORS.get(mode, None)
        hatch = MODE_HATCHES.get(mode, None)
        plt.fill_between(
            x,
            y - s,
            y + s,
            alpha=0.25,
            label=mode,
            facecolor=color,
            edgecolor=color,
            linewidth=1.0,
            hatch=hatch,
            zorder=2,
        )
    plt.xlabel("step")
    plt.ylabel(metric)
    plt.title(f"{metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_comparison_grid(agg_per_mode: Dict[str, pd.DataFrame], metrics: List[str], out_path: str):
    if not metrics:
        return
    # 动态网格：2 列，行数按需
    n = len(metrics)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4.5 * nrows))
    # 统一处理 axes 索引
    if isinstance(axes, (list, tuple)):
        flat_axes = [ax for ax in axes]
    else:
        try:
            flat_axes = axes.ravel()
        except Exception:
            flat_axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = flat_axes[idx]
        for mode, agg in agg_per_mode.items():
            if agg.empty:
                continue
            x = agg["step"].values
            y = agg[f"{metric}_mean"].values
            s = agg[f"{metric}_std"].values
            color = MODE_COLORS.get(mode, None)
            hatch = MODE_HATCHES.get(mode, None)
            ax.fill_between(
                x,
                y - s,
                y + s,
                alpha=0.25,
                label=mode,
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
                hatch=hatch,
                zorder=2,
            )
        ax.set_xlabel("step")
        # 去除与标题重复的信息，仅用标题展示精简指标名
        ax.set_ylabel("")
        ax.set_title(metric.replace("val/", ""))
        # 统一纵轴刻度数量（强制为 5 个主刻度）
        ax.yaxis.set_major_locator(LinearLocator(5))
        # 固定小数位，避免过多小数
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        ax.legend()

    # 处理多余子图（当 metrics 数少于网格容量）
    for j in range(idx + 1, len(flat_axes)):
        fig.delaxes(flat_axes[j])

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    agg_per_mode: Dict[str, pd.DataFrame] = {}
    metrics_available: List[str] = []

    for mode, pattern in MODES.items():
        files = find_metrics_files(pattern)
        print(f"Mode {mode}: found {len(files)} seeds")
        agg, val_cols = aggregate_mode(mode, files)
        agg_per_mode[mode] = agg
        if not metrics_available and not agg.empty:
            metrics_available = val_cols

    # Align across modes by common steps, then fallback to shortest length by index
    non_empty_aggs = [df for df in agg_per_mode.values() if not df.empty]
    if non_empty_aggs:
        common_steps = set(non_empty_aggs[0]["step"].tolist())
        for df in non_empty_aggs[1:]:
            common_steps &= set(df["step"].tolist())
        if common_steps:
            common_steps = sorted(common_steps)
            for mode, df in list(agg_per_mode.items()):
                if df.empty:
                    continue
                aligned = df[df["step"].isin(common_steps)].sort_values("step").reset_index(drop=True)
                agg_per_mode[mode] = aligned
        else:
            # fallback: truncate all to the shortest length by index
            min_len = min(len(df) for df in non_empty_aggs)
            for mode, df in list(agg_per_mode.items()):
                if df.empty:
                    continue
                agg_per_mode[mode] = df.iloc[:min_len].reset_index(drop=True)

    MIN_START_STEP = 1000
    for mode, df in list(agg_per_mode.items()):
        if df.empty:
            continue
        df = df[df["step"] >= MIN_START_STEP].reset_index(drop=True)
        agg_per_mode[mode] = df
        out_csv = os.path.join(OUTPUT_DIR, f"{mode}_agg.csv")
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

    # choose a set of typical metrics for plots if available
    preferred = [
        "val/ndcg@5",
        "val/ndcg@10",
        "val/recall@5",
        "val/recall@10",
    ]
    plot_metrics = [m for m in preferred if m in metrics_available] or metrics_available[:1]

    # 将多张图合并为一张多子图图片
    out_png = os.path.join(OUTPUT_DIR, "compare_multi.png")
    plot_comparison_grid(agg_per_mode, plot_metrics, out_png)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()


