# -*- coding: utf-8 -*-
r"""
建立標注工作區:複製篩選後照片 + 用 best.pt 產生 YOLO 格式預標注。

直接執行:python gen_prelabels.py
輸出:辨識模型\真實貨櫃外觀\標注工作區\
    ├── classes.txt        (0=container, 1=damage)
    ├── <照片>.jpg/.png    (自 篩選後\有破、沒破 複製)
    └── <照片>.txt         (模型預標注,同名;人工修完即為訓練標注)
"""
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "runs" / "damage_yolo11s" / "weights" / "best.pt"
SRC = HERE.parent / "真實貨櫃外觀" / "篩選後"
WORK = HERE.parent / "真實貨櫃外觀" / "標注工作區"
CONF = 0.15
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    from ultralytics import YOLO

    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "classes.txt").write_text("container\ndamage\n", encoding="utf-8")
    model = YOLO(str(WEIGHTS))

    n_img = n_box = 0
    for sub in ("有破", "沒破"):
        for p in sorted((SRC / sub).iterdir()):
            if p.suffix.lower() not in IMG_EXT:
                continue
            dst = WORK / p.name
            if not dst.exists():
                shutil.copy2(p, dst)

            r = model.predict(str(p), conf=CONF, imgsz=960, verbose=False)[0]
            h, w = r.orig_shape
            lines = []
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{int(b.cls[0])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                n_box += 1
            dst.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            n_img += 1

    print(f"完成:{n_img} 張照片、{n_box} 個預標注框 → {WORK}")
    print("人工修框重點:刪錯框、補漏框(特別是凹陷)、框貼緊破損邊緣")


if __name__ == "__main__":
    main()
