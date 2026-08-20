# -*- coding: utf-8 -*-
"""
用 best.pt 在合成 val 上做進階評估:

1. 圖層級「漏櫃率」:有破損的圖,模型是否至少框到一個 damage
2. 按原始破損類別(axis/concave/dentado/perforation)的框召回率(IoU>=0.5)
3. 把「誤報 damage」(與任何 GT 破損 IoU<0.1)裁圖拼成 fp_grid.png,
   用來檢查是否把塗鴉/標誌誤判成破損

直接執行:python eval_missrate.py
產出:runs/damage_yolo11s/eval_missrate.txt、fp_grid.png
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

DATASET = Path(r"D:\Arnold_chun_yen\container_damage\dataset")
COCO_VAL = Path(r"D:\Arnold_chun_yen\SeaFront_v1_0_0\val.json")
RUN = Path(__file__).resolve().parent / "runs" / "damage_yolo11s"
WEIGHTS = RUN / "weights" / "best.pt"
CONF = 0.25          # 與 ultralytics 預設一致
IOU_HIT = 0.50       # GT 框被視為「有偵測到」的門檻
IOU_FP = 0.10        # 預測框與所有 GT 破損重疊皆低於此 → 視為誤報
MAX_FP_CROPS = 48
ID2NAME = {1: "axis", 2: "concave", 3: "dentado", 4: "perforation"}


def iou(a, b):
    """a, b: [x1, y1, x2, y2]"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def main():
    from ultralytics import YOLO

    # GT:從 COCO val.json 取原始 4 類破損框(轉 xyxy 像素座標)
    coco = json.loads(COCO_VAL.read_text(encoding="utf-8"))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    gt = defaultdict(list)   # file_name -> [(orig_cls, [x1,y1,x2,y2]), ...]
    for a in coco["annotations"]:
        if a["category_id"] == 0:
            continue                       # 略過 container
        x, y, w, h = a["bbox"]
        gt[id2file[a["image_id"]]].append((a["category_id"], [x, y, x + w, y + h]))

    val_imgs = sorted((DATASET / "images" / "val").glob("*.png"))
    print(f"val 影像 {len(val_imgs)} 張,載入權重 {WEIGHTS.name} ...")
    model = YOLO(str(WEIGHTS))

    img_cm = Counter()                         # 圖層級 TP/FN/FP/TN
    cls_total, cls_hit = Counter(), Counter()  # 各原始類別框召回
    fp_crops = []                              # (crop, conf, fname)

    for i, img_path in enumerate(val_imgs, 1):
        r = model.predict(str(img_path), conf=CONF, imgsz=640, verbose=False)[0]
        if i % 400 == 0:
            print(f"  進度 {i}/{len(val_imgs)}")
        fname = Path(r.path).name
        preds = [b.xyxy[0].tolist() + [float(b.conf[0])]
                 for b in r.boxes if int(b.cls[0]) == 1]      # 只取 damage 類
        gts = gt.get(fname, [])

        # 1) 圖層級混淆
        key = ("TP" if preds else "FN") if gts else ("FP" if preds else "TN")
        img_cm[key] += 1

        # 2) 各原始類別框召回
        for cls_id, gbox in gts:
            cls_total[cls_id] += 1
            if any(iou(gbox, p[:4]) >= IOU_HIT for p in preds):
                cls_hit[cls_id] += 1

        # 3) 誤報裁圖(與所有 GT 破損幾乎不重疊的預測框)
        for p in preds:
            if all(iou(g[1], p[:4]) < IOU_FP for g in gts) and len(fp_crops) < MAX_FP_CROPS:
                im = Image.open(r.path).convert("RGB")
                x1, y1, x2, y2 = [int(v) for v in p[:4]]
                pad = 30
                crop = im.crop((max(0, x1 - pad), max(0, y1 - pad),
                                min(im.width, x2 + pad), min(im.height, y2 + pad)))
                fp_crops.append((crop, p[4], fname))

    tp, fn, fp, tn = img_cm["TP"], img_cm["FN"], img_cm["FP"], img_cm["TN"]
    lines = [
        f"=== 圖層級(conf>={CONF})===",
        f"有破損的圖:{tp + fn}(抓到 {tp},漏掉 {fn})→ 漏櫃率 {fn / max(1, tp + fn):.2%}",
        f"無破損的圖:{fp + tn}(誤報 {fp})→ 誤報率 {fp / max(1, fp + tn):.2%}",
        "",
        f"=== 各原始類別框召回(IoU>={IOU_HIT})===",
    ]
    for cid in sorted(cls_total):
        t, h = cls_total[cid], cls_hit[cid]
        lines.append(f"{ID2NAME[cid]:12s}: {h}/{t} = {h / t:.2%}")
    report = "\n".join(lines)
    print("\n" + report)
    (RUN / "eval_missrate.txt").write_text(report, encoding="utf-8")

    if fp_crops:
        cell, cols = 220, 8
        rows = (len(fp_crops) + cols - 1) // cols
        grid = Image.new("RGB", (cols * cell, rows * cell), (30, 30, 30))
        dr = ImageDraw.Draw(grid)
        for i, (crop, conf, fname) in enumerate(fp_crops):
            crop.thumbnail((cell - 8, cell - 8))
            x, y = (i % cols) * cell, (i // cols) * cell
            grid.paste(crop, (x + 4, y + 4))
            dr.text((x + 6, y + 4), f"{conf:.2f}", fill=(255, 80, 80))
        grid.save(RUN / "fp_grid.png")
        print(f"\n誤報裁圖 {len(fp_crops)} 個 → {RUN / 'fp_grid.png'}")
    else:
        print("\n沒有任何誤報框(與 GT 破損完全不重疊的預測)")


if __name__ == "__main__":
    main()
