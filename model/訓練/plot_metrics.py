# -*- coding: utf-8 -*-
"""
從 ultralytics 的 results.csv 畫「Loss + Macro F1」合併圖(一張 PNG、兩個子圖)。

用法:
    python plot_metrics.py               ← 自動抓 runs/ 下最新一次訓練
    python plot_metrics.py <run資料夾>   ← 指定訓練輸出資料夾

Macro F1 說明:ultralytics 的 precision/recall 欄位是「各類別平均」(macro),
故 Macro F1 = 2*P*R/(P+R) 以該兩欄計算。
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS_DIR = Path(__file__).resolve().parent / "runs"


def make_plot(run_dir: Path) -> Path:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 {csv_path}")

    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k.strip(): v.strip() for k, v in row.items() if k})

    def col(name):
        return [float(r[name]) for r in rows]

    epochs = [int(float(r["epoch"])) for r in rows]
    train_loss = [a + b + c for a, b, c in
                  zip(col("train/box_loss"), col("train/cls_loss"), col("train/dfl_loss"))]
    val_loss = [a + b + c for a, b, c in
                zip(col("val/box_loss"), col("val/cls_loss"), col("val/dfl_loss"))]
    precision = col("metrics/precision(B)")
    recall = col("metrics/recall(B)")
    macro_f1 = [2 * p * r / (p + r) if (p + r) > 0 else 0.0
                for p, r in zip(precision, recall)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, train_loss, label="train loss", color="#1f77b4")
    ax1.plot(epochs, val_loss, label="val loss", color="#d62728")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss (box+cls+dfl)")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    best_i = max(range(len(macro_f1)), key=lambda i: macro_f1[i])
    ax2.plot(epochs, macro_f1, label="Macro F1", color="#2ca02c")
    ax2.scatter([epochs[best_i]], [macro_f1[best_i]], color="#2ca02c", zorder=3)
    ax2.annotate(f"best {macro_f1[best_i]:.3f} @ep{epochs[best_i]}",
                 (epochs[best_i], macro_f1[best_i]),
                 textcoords="offset points", xytext=(8, -12), fontsize=9)
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("Macro F1")
    ax2.set_ylim(0, 1)
    ax2.set_title("Macro F1 (val)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(run_dir.name)
    fig.tight_layout()
    out = run_dir / "loss_macroF1.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def latest_run(runs_dir: Path) -> Path:
    candidates = [d for d in runs_dir.iterdir()
                  if d.is_dir() and (d / "results.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"{runs_dir} 下找不到含 results.csv 的訓練輸出")
    return max(candidates, key=lambda d: (d / "results.csv").stat().st_mtime)


if __name__ == "__main__":
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run(RUNS_DIR)
    print("畫圖來源:", run)
    print("已輸出:", make_plot(run))
