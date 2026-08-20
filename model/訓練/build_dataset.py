# -*- coding: utf-8 -*-
"""
把 SeaFront 轉成 2 類 YOLO 偵測資料集(0=container, 1=damage)。

直接執行即可,不需參數:
    python build_dataset.py

- 圖片用硬連結(同在 D 槽,不佔額外空間);失敗自動改用複製
- 標注來源 bbannotation/(5 類),類別 1~4 全部併成 1=damage
- 沿用官方 train/val 切分,唯一跨切分的場景整組歸 train(避免洩漏)
- 產出 data.yaml,可直接餵 ultralytics
"""
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

SRC = Path(r"D:\Arnold_chun_yen\SeaFront_v1_0_0")
DST = Path(r"D:\Arnold_chun_yen\container_damage\dataset")


def main():
    if not SRC.exists():
        print("找不到 SeaFront 資料集:", SRC)
        sys.exit(1)

    # 1) 收集官方切分,場景 = 檔名的時間戳前綴
    split_files = {s: sorted(os.listdir(SRC / "images" / s)) for s in ("train", "val")}
    scenes = {s: {f.split("_")[0] for f in fs} for s, fs in split_files.items()}
    overlap = scenes["train"] & scenes["val"]
    print(f"官方切分:train {len(split_files['train'])} 張 / val {len(split_files['val'])} 張;"
          f"跨切分場景 {len(overlap)} 個 → 整組歸 train")

    # 2) 建輸出結構
    for s in ("train", "val"):
        (DST / "images" / s).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / s).mkdir(parents=True, exist_ok=True)

    def link_or_copy(src: Path, dst: Path):
        if dst.exists():
            return
        try:
            os.link(src, dst)          # 同磁碟硬連結,零空間成本
        except OSError:
            shutil.copy2(src, dst)

    # 3) 逐張處理
    stats = Counter()
    box_stats = Counter()
    for split, files in split_files.items():
        for fname in files:
            scene = fname.split("_")[0]
            final = "train" if scene in overlap else split   # 重疊場景強制歸 train
            stem = fname.rsplit(".", 1)[0]

            link_or_copy(SRC / "images" / split / fname, DST / "images" / final / fname)

            src_label = SRC / "bbannotation" / f"{stem}.txt"
            lines_out = []
            if src_label.exists():
                for line in src_label.read_text().strip().splitlines():
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    cls = 0 if parts[0] == "0" else 1        # 0=container,其餘併成 damage
                    lines_out.append(" ".join([str(cls)] + parts[1:]))
                    box_stats[f"{final}/{'container' if cls == 0 else 'damage'}"] += 1
            else:
                stats["missing_label"] += 1
            (DST / "labels" / final / f"{stem}.txt").write_text("\n".join(lines_out) + "\n")
            stats[f"img_{final}"] += 1

    # 4) data.yaml
    (DST / "data.yaml").write_text(
        f"path: {DST.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: container\n  1: damage\n",
        encoding="utf-8",
    )

    print(f"\n完成!輸出於 {DST}")
    print(f"  圖片:train {stats['img_train']} 張 / val {stats['img_val']} 張"
          f"(缺標注 {stats['missing_label']} 張)")
    for k in sorted(box_stats):
        print(f"  框數 {k}: {box_stats[k]}")
    print("  data.yaml 已寫出,可直接用於訓練")


if __name__ == "__main__":
    main()
