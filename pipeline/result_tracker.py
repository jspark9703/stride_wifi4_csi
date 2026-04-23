"""
pipeline/result_tracker.py
===========================
실험 결과 저장 및 Ablation Study 비교 집계.
"""

import os
import csv
import json
import copy

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG    = "#0D1117"
PANEL = "#161B22"
TEXT  = "#E6EDF3"
GRID  = "#30363D"

plt.rcParams.update({
    "figure.facecolor": BG,  "axes.facecolor": PANEL,
    "axes.edgecolor":   GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color":  TEXT, "grid.color":  GRID,
    "font.family": "DejaVu Sans", "font.size": 10,
})


# ──────────────────────────────────────────────────────────────
#  단일 실험 결과 저장
# ──────────────────────────────────────────────────────────────

def save_run(run_dir: str, cfg: dict, metrics: dict) -> None:
    """실험 설정과 지표를 JSON으로 저장."""
    os.makedirs(run_dir, exist_ok=True)

    # config 복사본 저장 (재현성)
    import yaml
    with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)

    # 지표 저장
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────
#  Ablation 요약 CSV 누적 저장
# ──────────────────────────────────────────────────────────────

def append_summary(summary_path: str, record: dict) -> None:
    """ablation_summary.csv에 한 행 추가."""
    write_header = not os.path.exists(summary_path)
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(record)


# ──────────────────────────────────────────────────────────────
#  Ablation 비교 그래프
# ──────────────────────────────────────────────────────────────

def plot_ablation(summary_path: str, param_name: str, out_dir: str) -> None:
    """파라미터 값 vs 정확도 라인 그래프 생성."""
    if not os.path.exists(summary_path):
        return

    records = []
    with open(summary_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    if not records:
        return

    x_labels = [str(r.get("param_value", r.get("exp_name", i)))
                 for i, r in enumerate(records)]
    y_acc = [float(r["test_accuracy"]) for r in records]
    y_cv  = [float(r["cv_mean"])       for r in records]
    y_std = [float(r["cv_std"])        for r in records]

    fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 1.2), 5), facecolor=BG)
    ax.set_facecolor(PANEL)
    ax.spines[:].set_color(GRID)
    ax.tick_params(colors=TEXT)

    xs = range(len(x_labels))
    ax.plot(xs, y_acc, "o-",  color="#EF5350", label="Test Accuracy",
            linewidth=2, markersize=7, zorder=3)
    ax.plot(xs, y_cv,  "s--", color="#42A5F5", label="CV Mean",
            linewidth=1.5, markersize=5, zorder=3)
    ax.fill_between(
        xs,
        [cv - sd for cv, sd in zip(y_cv, y_std)],
        [cv + sd for cv, sd in zip(y_cv, y_std)],
        color="#42A5F5", alpha=0.2, label="CV ± Std",
    )

    # 최고점 강조
    best_i = int(np.argmax(y_acc))
    ax.annotate(
        f"best: {y_acc[best_i]:.4f}",
        xy=(best_i, y_acc[best_i]),
        xytext=(best_i, y_acc[best_i] + 0.01),
        ha="center", color="#EF5350", fontsize=9,
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, color=TEXT, rotation=15, ha="right")
    ax.set_xlabel(param_name, color=TEXT)
    ax.set_ylabel("Accuracy", color=TEXT)
    ax.set_title(f"Ablation: {param_name}", color=TEXT, fontsize=12, pad=10)
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    ax.set_ylim(max(0, min(y_acc) - 0.05), min(1.0, max(y_acc) + 0.05))

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "comparison_plot.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[Tracker] Plot → {out_path}")


# ──────────────────────────────────────────────────────────────
#  콘솔 요약 출력
# ──────────────────────────────────────────────────────────────

def print_summary(records: list, param_name: str) -> None:
    if not records:
        return

    header = f"{'실험명':<35} {'파라미터값':>15} {'Test Acc':>10} {'CV Mean':>10} {'CV Std':>8}"
    print(f"\n{'─'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")

    best = max(records, key=lambda r: float(r.get("test_accuracy", 0)))

    for r in records:
        mark = " ★" if r is best else ""
        print(
            f" {r.get('exp_name',''):<34} "
            f"{str(r.get('param_value',''))!s:>15} "
            f"{float(r.get('test_accuracy',0)):>10.4f} "
            f"{float(r.get('cv_mean',0)):>10.4f} "
            f"{float(r.get('cv_std',0)):>8.4f}"
            f"{mark}"
        )

    print(f"{'─'*len(header)}")
    print(
        f"  Best → param={best.get('param_value')}  "
        f"acc={float(best.get('test_accuracy',0)):.4f}  "
        f"cv={float(best.get('cv_mean',0)):.4f}±{float(best.get('cv_std',0)):.4f}"
    )
