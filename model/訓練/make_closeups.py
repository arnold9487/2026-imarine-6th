# -*- coding: utf-8 -*-
r"""
從已建好的 2 類資料集(train split)生成「特寫」子集,縮小全景→特寫的尺度 gap。

直接執行:python make_closeups.py

原理:合成圖裡每個 damage 框,以隨機倍率(框長邊的 1.6~5 倍)裁一個窗口,
damage 在窗口內佔比大幅提高,等於免費製造特寫訓練樣本;標注自動換算。

- 只從 train split 生成(val 不動,避免洩漏)
- 每個 damage 實例生成 2 張裁切
- 輸出:dataset\images\closeup + dataset\labels\closeup,並改寫 data.yaml
  讓 train 同時吃 [images/train, images/closeup]
- 另存 9 張畫框抽樣圖到本資料夾 closeup_check.jpg 供人工檢查
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw

DST = Path(r"D:\Arnold_chun_yen\container_damage\dataset")
CROPS_PER_BOX = 2          # 每個 damage 實例生成幾張特寫
MARGIN_RANGE = (1.6, 5.0)  # 裁切窗口 = damage 框長邊 × 此倍率
MIN_SIDE = 256             # 窗口最小邊長(px),避免放大後過度模糊
JPEG_Q = 92
SEED = 0


def load_boxes(label_path, W, H):
    """讀 YOLO txt → [(cls, x1, y1, x2, y2)](像素座標)"""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) != 5:
            continue
        cls = int(p[0])
        cx, cy, w, h = (float(v) for v in p[1:])
        boxes.append((cls, (cx - w / 2) * W, (cy - h / 2) * H,
                      (cx + w / 2) * W, (cy + h / 2) * H))
    return boxes


def crop_window(box, W, H, rng):
    """為一個 damage 框選一個包含它的隨機裁切窗口,回傳 (x1,y1,x2,y2) 或 None"""
    _, bx1, by1, bx2, by2 = box
    long_side = max(bx2 - bx1, by2 - by1)
    size = long_side * rng.uniform(*MARGIN_RANGE)
    ar = rng.uniform(0.8, 1.25)
    cw = max(min(size * ar, W), 1)
    ch = max(min(size / ar, H), 1)
    if max(cw, ch) < MIN_SIDE:
        scale = MIN_SIDE / max(cw, ch)
        cw, ch = cw * scale, ch * scale
    cw, ch = min(cw, W), min(ch, H)
    # 窗口需完整包含 damage 框:x1 ∈ [bx2-cw, bx1] ∩ [0, W-cw]
    x_lo, x_hi = max(0.0, bx2 - cw), min(bx1, W - cw)
    y_lo, y_hi = max(0.0, by2 - ch), min(by1, H - ch)
    if x_lo > x_hi or y_lo > y_hi:   # 框比窗口大(幾乎不會發生)→ 放棄
        return None
    x1 = rng.uniform(x_lo, x_hi)
    y1 = rng.uniform(y_lo, y_hi)
    return x1, y1, x1 + cw, y1 + ch


def boxes_in_crop(boxes, win):
    """換算窗口內的標注:damage 需 ≥60% 面積在窗內;container 允許被裁(特寫本來就看不到全櫃)"""
    wx1, wy1, wx2, wy2 = win
    cw, ch = wx2 - wx1, wy2 - wy1
    out = []
    for cls, x1, y1, x2, y2 in boxes:
        ix1, iy1 = max(x1, wx1), max(y1, wy1)
        ix2, iy2 = min(x2, wx2), min(y2, wy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter = (ix2 - ix1) * (iy2 - iy1)
        own = (x2 - x1) * (y2 - y1)
        keep = inter >= 0.6 * own if cls == 1 else inter >= 0.5 * min(own, cw * ch)
        if not keep:
            continue
        out.append((cls, ((ix1 + ix2) / 2 - wx1) / cw, ((iy1 + iy2) / 2 - wy1) / ch,
                    (ix2 - ix1) / cw, (iy2 - iy1) / ch))
    return out


def main():
    rng = random.Random(SEED)
    img_dir = DST / "images" / "train"
    lbl_dir = DST / "labels" / "train"
    out_img = DST / "images" / "closeup"
    out_lbl = DST / "labels" / "closeup"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    n_crops = n_src = 0
    samples = []
    for img_path in sorted(img_dir.iterdir()):
        boxes = load_boxes(lbl_dir / f"{img_path.stem}.txt", 1920, 1080)
        dmg = [b for b in boxes if b[0] == 1]
        if not dmg:
            continue
        img = Image.open(img_path)
        W, H = img.size
        if (W, H) != (1920, 1080):   # 保險:實際尺寸不同就重讀框
            boxes = load_boxes(lbl_dir / f"{img_path.stem}.txt", W, H)
            dmg = [b for b in boxes if b[0] == 1]
        n_src += 1
        for bi, box in enumerate(dmg):
            for ci in range(CROPS_PER_BOX):
                win = crop_window(box, W, H, rng)
                if win is None:
                    continue
                new_boxes = boxes_in_crop(boxes, win)
                if not any(b[0] == 1 for b in new_boxes):
                    continue
                name = f"{img_path.stem}_z{bi}{ci}"
                crop = img.crop(tuple(round(v) for v in win))
                crop.convert("RGB").save(out_img / f"{name}.jpg", quality=JPEG_Q)
                (out_lbl / f"{name}.txt").write_text(
                    "\n".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                              for c, cx, cy, w, h in new_boxes) + "\n")
                n_crops += 1
                if len(samples) < 9 and rng.random() < 0.02:
                    samples.append((crop.copy(), new_boxes))

    # data.yaml:train 同時吃原圖 + 特寫
    (DST / "data.yaml").write_text(
        f"path: {DST.as_posix()}\n"
        "train:\n  - images/train\n  - images/closeup\n"
        "val: images/val\n"
        "names:\n  0: container\n  1: damage\n",
        encoding="utf-8",
    )

    # 抽樣畫框檢查圖(3×3)
    if samples:
        cell = 320
        grid = Image.new("RGB", (cell * 3, cell * 3), (30, 30, 30))
        for i, (im, bs) in enumerate(samples):
            im = im.resize((cell, cell))
            d = ImageDraw.Draw(im)
            for c, cx, cy, w, h in bs:
                x1, y1 = (cx - w / 2) * cell, (cy - h / 2) * cell
                x2, y2 = (cx + w / 2) * cell, (cy + h / 2) * cell
                d.rectangle([x1, y1, x2, y2],
                            outline=(255, 80, 80) if c == 1 else (80, 160, 255), width=3)
            grid.paste(im, ((i % 3) * cell, (i // 3) * cell))
        grid.save(Path(__file__).resolve().parent / "closeup_check.jpg", quality=90)

    print(f"完成:{n_src} 張含破損原圖 → {n_crops} 張特寫")
    print(f"輸出:{out_img}")
    print("data.yaml 已改寫(train = images/train + images/closeup)")
    print("抽樣檢查圖:closeup_check.jpg(紅=damage、藍=container)")


if __name__ == "__main__":
    main()
