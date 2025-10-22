import os
import glob
from typing import Dict, List, Tuple
import math

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


BASE_DIR = "/data/zhantianyu/Project/GRID/logs/train/runs/toys/gr"
OUTPUT_DIR = os.path.join(BASE_DIR, "aggregates")


MODES = {
    "base": "rkmeans_original_seed_*",
    "rastp_l1": "rkmeans_rastp_l1_seed_*",
    "rastp_l2": "rkmeans_rastp_l2_seed_*",
}

# 固定颜色与纹理，增强模式区分度
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


def find_metrics_files(pattern: str) -> Dict[str, str]:
    seed_dirs = glob.glob(os.path.join(BASE_DIR, pattern))
    results: Dict[str, str] = {}
    for seed_dir in seed_dirs:
        csv_path = os.path.join(seed_dir, "csv", "version_0", "metrics.csv")
        if os.path.isfile(csv_path):
            results[os.path.basename(seed_dir)] = csv_path
    return results


def load_seed_test_row(csv_path: str) -> pd.Series:
    df = pd.read_csv(csv_path)
    test_cols = [c for c in df.columns if c.startswith("test/")]
    # 过滤掉包含 loss 的测试指标
    test_cols = [c for c in test_cols if "loss" not in c.lower()]
    if not test_cols:
        return pd.Series(dtype=float)
    # 选择最后一步（step 最大）的测试结果
    if "step" in df.columns:
        df = df.sort_values("step").drop_duplicates(subset=["step"], keep="last")
        last_row = df.iloc[-1]
    else:
        last_row = df.iloc[-1]
    return last_row[test_cols]


def aggregate_mode(files: Dict[str, str]) -> Tuple[pd.Series, pd.Series, List[str]]:
    # 逐 seed 读取最后一步的 test 指标
    rows: List[pd.Series] = []
    for seed_name, path in files.items():
        try:
            s = load_seed_test_row(path)
            if not s.empty:
                rows.append(s)
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    if not rows:
        return pd.Series(dtype=float), pd.Series(dtype=float), []

    df = pd.DataFrame(rows)
    # 对齐所有出现过的 test 列，按列计算均值/方差（自动 skipna）
    means = df.mean(axis=0)
    stds = df.std(axis=0, ddof=1)
    metrics = list(df.columns)
    return means, stds, metrics


def plot_grouped_bars(stats_per_mode: Dict[str, Tuple[pd.Series, pd.Series]], metrics: List[str], out_path: str):
    modes = list(stats_per_mode.keys())
    num_metrics = len(metrics)
    num_modes = len(modes)

    if num_metrics == 0 or num_modes == 0:
        print("No data to plot.")
        return

    # 风格与全局美化
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except Exception:
        plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "font.weight": "regular"
    })

    bar_width = 0.8 / max(num_modes, 1)
    x = range(num_metrics)

    plt.figure(figsize=(max(7, num_metrics * 1.5), 5.5))

    global_min = float("inf")
    global_max = float("-inf")

    bar_containers = []
    for i, mode in enumerate(modes):
        means, stds = stats_per_mode[mode]
        y = [means.get(m, float("nan")) for m in metrics]
        e = [stds.get(m, float("nan")) for m in metrics]

        offset = (i - (num_modes - 1) / 2) * bar_width
        positions = [xi + offset for xi in x]
        color = MODE_COLORS.get(mode, None)
        hatch = MODE_HATCHES.get(mode, None)

        bars = plt.bar(
            positions,
            y,
            yerr=e,
            width=bar_width * 0.9,
            capsize=4,
            label=mode.replace('_', ' ').title(),
            color=color,
            edgecolor="#2f2f2f",
            hatch=hatch,
            linewidth=1.0,
            zorder=2,
        )
        bar_containers.append((bars, y))

        for yy, ee in zip(y, e):
            if not (isinstance(yy, float) and math.isnan(yy)):
                low = yy - (ee if not (isinstance(ee, float) and math.isnan(ee)) else 0.0)
                high = yy + (ee if not (isinstance(ee, float) and math.isnan(ee)) else 0.0)
                if low < global_min:
                    global_min = low
                if high > global_max:
                    global_max = high

    if global_min == float("inf") or global_max == float("-inf"):
        global_min, global_max = 0.0, 1.0
    rng = max(global_max - global_min, 1e-6)
    margin = 0.05 * rng
    plt.ylim(global_min - margin, global_max + margin)

    plt.xticks(list(x), [m.replace("test/", "") for m in metrics], rotation=20, ha='right')
    plt.ylabel("Score", fontweight='bold')
    plt.title("Test Results", fontweight='bold')

    # 图例放在图内，竖着（单列），加粗文字
    legend = plt.legend(
        ncol=1,                     # 竖排
        frameon=False,
        loc='upper left',          # 图内右上角（可选：'lower right', 'upper left' 等）
        prop={'weight': 'bold'}     # 图例文字加粗
    )

    plt.tight_layout()
    plt.grid(axis="y", alpha=0.2, linestyle="--", zorder=1)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # 柱顶数值标签更大
    for bars, y in bar_containers:
        for rect, yy in zip(bars, y):
            if isinstance(yy, float) and math.isnan(yy):
                continue
            height = rect.get_height()
            ax.annotate(f"{yy:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 12),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=12,
                        color="#333333",
                        zorder=4,
                        clip_on=False)

    plt.savefig(out_path, dpi=220, bbox_inches='tight', pad_inches=0.01)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stats_per_mode: Dict[str, Tuple[pd.Series, pd.Series]] = {}
    global_metrics: List[str] = []

    for mode, pattern in MODES.items():
        files = find_metrics_files(pattern)
        print(f"Mode {mode}: found {len(files)} seeds")
        means, stds, metrics = aggregate_mode(files)
        stats_per_mode[mode] = (means, stds)
        # 记录所有出现过的 test 指标，保持顺序稳定
        for m in metrics:
            if m not in global_metrics:
                global_metrics.append(m)

        # 导出该模式的统计表
        if len(metrics) > 0:
            out_csv = os.path.join(OUTPUT_DIR, f"{mode}_test_stats.csv")
            pd.DataFrame({
                "metric": metrics,
                "mean": [means.get(m, float("nan")) for m in metrics],
                "std": [stds.get(m, float("nan")) for m in metrics],
            }).to_csv(out_csv, index=False)
            print(f"Saved: {out_csv}")

    if not global_metrics:
        print("No test metrics found.")
        return

    # 将 @5 放在左侧的优先顺序；其余保持原有顺序附加在后
    preferred = [
        "test/ndcg@5",
        "test/ndcg@10",
        "test/recall@5",
        "test/recall@10",
    ]
    ordered_metrics = [m for m in preferred if m in global_metrics] + [m for m in global_metrics if m not in preferred]

    out_png = os.path.join(OUTPUT_DIR, "test_bars.png")
    plot_grouped_bars(stats_per_mode, ordered_metrics, out_png)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()


