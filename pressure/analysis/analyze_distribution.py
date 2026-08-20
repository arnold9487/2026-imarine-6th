# -*- coding: utf-8 -*-
"""分析各 CT「每日最大同時在泊艘次（小船等量）」分布（2025 全年、僅營運日）
- 壓力上限 = 營運日第 95 百分位（不假設分布，避免 Poisson 爭議）
- 圖上另標第 99 百分位（極端負荷參考）
- 輸出 distribution.png（3x3：7 CT + 統計表 + 方法說明）
"""
import sys, os, json, math
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- 中文字型（Windows 預設黑體）----
for f in ("C:/Windows/Fonts/msjh.ttc", "C:/Windows/Fonts/msyh.ttc"):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "sans-serif"]
        break
plt.rcParams["axes.unicode_minus"] = False

# ---- 讀資料 ----
DATA_DIR = os.path.normpath(os.path.join(ROOT, "..", "data"))
data = json.load(open(os.path.join(DATA_DIR, "daymax.json"), encoding="utf-8"))
port = json.load(open(os.path.join(DATA_DIR, "port_data.json"), encoding="utf-8"))
CTS = ["CT1", "CT2", "CT3", "CT4", "CT5", "CT6", "CT7"]
NAMES = {ct: port["terminals"][ct]["name_zh"] for ct in CTS}

fig, axs = plt.subplots(3, 3, figsize=(15, 11))
axs = axs.flatten()

summary_rows = []
plot_data = []
for ct in CTS:
    raw = np.array(data[ct], dtype=int)
    nz = raw[raw > 0]
    n = len(nz); mean = nz.mean(); var = nz.var(ddof=0)
    p95 = int(math.ceil(np.percentile(nz, 95)))
    p99 = int(math.ceil(np.percentile(nz, 99)))
    mx = int(nz.max())
    summary_rows.append((ct, n, len(raw) - n, mean, var, p95, p99, mx))
    plot_data.append({"ct": ct, "raw": raw, "nz": nz, "n": n, "mean": mean,
                      "var": var, "p95": p95, "p99": p99, "mx": mx})

# 全域尺度
x_max = max(max(p["nz"].max(), p["p99"]) for p in plot_data) + 1
y_max = 0
for p in plot_data:
    counts, _ = np.histogram(p["nz"], bins=np.arange(0.5, x_max + 1.5, 1))
    y_max = max(y_max, counts.max())
y_max = int(y_max * 1.08)
GLOBAL_BINS = np.arange(0.5, x_max + 1.5, 1)

for i, p in enumerate(plot_data):
    ax = axs[i]; ct = p["ct"]; nz = p["nz"]
    ax.hist(nz, bins=GLOBAL_BINS, edgecolor="#444", color="#7ba6dc", alpha=.85, label="營運日分布")
    ax.axvline(p["p95"], color="#d83a2c", linestyle="--", lw=1.6, label=f"第95%(上限)={p['p95']}")
    ax.axvline(p["p99"], color="#e0913a", linestyle=":",  lw=1.6, label=f"第99%={p['p99']}")
    ax.set_title(f"{ct} {NAMES[ct]}\n"
                 f"μ={p['mean']:.2f}  σ²={p['var']:.2f}  "
                 f"(營運日 {p['n']}/{len(p['raw'])})", fontsize=10)
    ax.set_xlabel("每日最大同時在泊艘次（小船等量）"); ax.set_ylabel("天數")
    ax.set_xlim(0.5, x_max + 0.5)
    ax.set_ylim(0, y_max)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=.3)

# 第 8 格：統計表（欄位精簡、加寬；標題換行避免擁擠）
ax = axs[7]; ax.axis("off")
header = ["CT", "營運日", "零日", "μ", "σ²", "第95%\n(上限)", "第99%", "最大值"]
cell = [[r[0], str(r[1]), str(r[2]), f"{r[3]:.2f}", f"{r[4]:.2f}",
         str(r[5]), str(r[6]), str(r[7])] for r in summary_rows]
tab = ax.table(cellText=cell, colLabels=header, loc="center", cellLoc="center")
tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1.15, 1.9)
ax.set_title("各中心負荷統計（單位＝小船等量）", fontsize=11)

# 第 9 格：方法說明（取代原本的 Poisson 判定建議）
ax = axs[8]; ax.axis("off")
notes = [
    "方法說明：",
    "",
    "• 樣本：2025 全年、僅營運日（排除無船日）",
    "   頭尾多取月份僅供掃描線算跨年在泊，不進統計",
    "",
    "• 壓力上限 = 營運日第 95 百分位",
    "   達全年前 5% 忙碌程度即視為滿載（紅）",
    "",
    "• 不採 Poisson：本變數是每日『最大同時在泊』，",
    "   屬極值而非到港計數；且樣本量不支持 99.85%",
    "   分位。經驗百分位不假設分布，直觀且無爭議。",
]
ax.text(0.02, 0.98, "\n".join(notes), va="top", ha="left", fontsize=10)

plt.suptitle("各貨櫃中心 每日最大同時在泊（小船等量）分布 — 2025 全年・僅營運日",
             fontsize=14, y=0.995)
plt.tight_layout()
out = os.path.join(ROOT, "distribution.png")
plt.savefig(out, dpi=130, bbox_inches="tight")
print("輸出", out)
print("\n=== 統計摘要 ===")
print(f"{'CT':4} {'營運日':>5} {'零日':>4} {'μ':>6} {'σ²':>6} {'第95%':>5} {'第99%':>5} {'最大':>4}")
for r in summary_rows:
    print(f"{r[0]:4} {r[1]:>5} {r[2]:>4} {r[3]:>6.2f} {r[4]:>6.2f} {r[5]:>5} {r[6]:>5} {r[7]:>4}")
