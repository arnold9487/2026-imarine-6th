# -*- coding: utf-8 -*-
r"""
用 best.pt 對真實照片做零樣本推論,輸出畫好框的圖供人工檢視。

直接執行:python predict_real.py
輸入:辨識模型\真實貨櫃外觀\篩選後\{有破,沒破}
輸出:runs\zero_shot_real\{有破,沒破}\*.jpg(畫框圖)+ summary.txt
"""
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 用法:python predict_real.py [run名稱]   例:python predict_real.py damage_yolo11s_zoom
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else "damage_yolo11s"
WEIGHTS = HERE / "runs" / RUN_NAME / "weights" / "best.pt"
SRC = Path(r"D:\Arnold_chun_yen\真實貨櫃外觀") / "篩選後"
OUT = HERE / "runs" / ("zero_shot_real" if RUN_NAME == "damage_yolo11s"
                       else f"zero_shot_real_{RUN_NAME.removeprefix('damage_yolo11s_')}")
CONF = 0.15                      # 低門檻拉召回;之後可比較 0.25
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    stats = Counter()
    lines = []

    for sub in ("有破", "沒破"):
        src_dir = SRC / sub
        out_dir = OUT / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        imgs = [p for p in sorted(src_dir.iterdir()) if p.suffix.lower() in IMG_EXT]
        print(f"[{sub}] {len(imgs)} 張")

        for p in imgs:
            r = model.predict(str(p), conf=CONF, imgsz=960, verbose=False)[0]
            n_dmg = sum(1 for b in r.boxes if int(b.cls[0]) == 1)
            n_ctn = sum(1 for b in r.boxes if int(b.cls[0]) == 0)
            stats[f"{sub}_圖數"] += 1
            stats[f"{sub}_有damage框"] += 1 if n_dmg else 0
            stats[f"{sub}_有container框"] += 1 if n_ctn else 0
            lines.append(f"{sub}/{p.name}: container×{n_ctn}, damage×{n_dmg}")
            r.save(str(out_dir / f"{p.stem}_pred.jpg"))

    total_d, hit_d = stats["有破_圖數"], stats["有破_有damage框"]
    total_c, fp_c = stats["沒破_圖數"], stats["沒破_有damage框"]
    summary = [
        f"=== 零樣本(conf>={CONF}, imgsz=960)===",
        f"有破 {total_d} 張:模型至少框到一個 damage 的有 {hit_d} 張"
        f"({hit_d / max(1, total_d):.1%})",
        f"有破但 container 有被框到:{stats['有破_有container框']}/{total_d}",
        f"沒破 {total_c} 張:誤框 damage 的有 {fp_c} 張",
        "",
        "※ 這只代表「有沒有框到東西」,框得準不準要人工看輸出圖",
    ]
    print("\n" + "\n".join(summary))
    (OUT / "summary.txt").write_text("\n".join(summary + ["", "--- 逐張 ---"] + lines),
                                     encoding="utf-8")
    print(f"\n畫框圖與 summary.txt 已存到 {OUT}")


if __name__ == "__main__":
    main()
