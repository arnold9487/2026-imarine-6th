# -*- coding: utf-8 -*-
"""
把 SeaFront_v1_0_0_TEST 轉成 2 類 YOLO 評估集(0=container, 1=damage)。

直接執行:python build_testset.py

- 來源:D:\\Arnold_chun_yen\\SeaFront_v1_0_0_TEST\\SeaFront_v1_0_0_TEST
  (images\\val 2,480 張 + bbannotation\\ 5 類 YOLO txt)
- 輸出:D:\\Arnold_chun_yen\\container_damage\\dataset_test + data.yaml
- 圖片硬連結(同 D 槽,零空間成本);類別 1~4 併成 1=damage
- TEST 全部場景與 train/val 不重疊,是乾淨的最終考卷
"""
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

SRC = Path(r"D:\Arnold_chun_yen\SeaFront_v1_0_0_TEST\SeaFront_v1_0_0_TEST")
DST = Path(r"D:\Arnold_chun_yen\container_damage\dataset_test")


def main():
    img_src = SRC / "images" / "val"
    lbl_src = SRC / "bbannotation"
    if not img_src.exists():
        print("找不到 TEST 圖片資料夾:", img_src)
        sys.exit(1)

    (DST / "images" / "test").mkdir(parents=True, exist_ok=True)
    (DST / "labels" / "test").mkdir(parents=True, exist_ok=True)

    def link_or_copy(src: Path, dst: Path):
        if dst.exists():
            return
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    stats = Counter()
    for img in sorted(img_src.iterdir()):
        if img.suffix.lower() != ".png":
            continue
        link_or_copy(img, DST / "images" / "test" / img.name)
        stats["img"] += 1

        src_label = lbl_src / f"{img.stem}.txt"
        lines_out = []
        if src_label.exists():
            for line in src_label.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls = 0 if parts[0] == "0" else 1
                lines_out.append(" ".join([str(cls)] + parts[1:]))
                stats["container" if cls == 0 else "damage"] += 1
        else:
            stats["missing_label"] += 1
        (DST / "labels" / "test" / f"{img.stem}.txt").write_text(
            "\n".join(lines_out) + "\n")

    (DST / "data.yaml").write_text(
        f"path: {DST.as_posix()}\n"
        "train: images/test\n"   # ultralytics 要求有 train 欄位,指到同處即可(不會用來訓練)
        "val: images/test\n"
        "names:\n  0: container\n  1: damage\n",
        encoding="utf-8",
    )

    print(f"完成!輸出於 {DST}")
    print(f"  圖片 {stats['img']} 張(缺標注 {stats['missing_label']})")
    print(f"  框數 container {stats['container']} / damage {stats['damage']}")


if __name__ == "__main__":
    main()
