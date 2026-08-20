# -*- coding: utf-8 -*-
r"""
用指定權重對 SeaFront TEST 集(2 類)做正式評估 + 抽樣畫框。

用法:
    python eval_test.py                     # 用最新的 damage_yolo11s* run
    python eval_test.py damage_yolo11s2     # 指定 run 名稱

輸出:
    runs\<run名>\test_eval\   ← ultralytics val 產物(混淆矩陣、PR 曲線)
    runs\<run名>\test_eval\summary.txt
    runs\<run名>\test_eval\samples\*.jpg   ← 12 張抽樣畫框圖(demo 素材)
"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = Path(r"D:\Arnold_chun_yen\container_damage\dataset_test\data.yaml")
TEST_IMGS = Path(r"D:\Arnold_chun_yen\container_damage\dataset_test\images\test")
TEST_LBLS = Path(r"D:\Arnold_chun_yen\container_damage\dataset_test\labels\test")


def pick_run() -> Path:
    if len(sys.argv) > 1:
        return HERE / "runs" / sys.argv[1]
    runs = sorted((HERE / "runs").glob("damage_yolo11s*"),
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        raise SystemExit("runs 下找不到 damage_yolo11s* 訓練結果")
    return runs[-1]


def main():
    from ultralytics import YOLO

    run = pick_run()
    weights = run / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"找不到權重:{weights}")
    print("使用權重:", weights)

    model = YOLO(str(weights))

    # ---- 1) 正式評估(2,480 張,含混淆矩陣與 PR 曲線圖)----
    r = model.val(data=str(DATA), imgsz=640, device=0,
                  project=str(run), name="test_eval", exist_ok=True)
    names = ["container", "damage"]
    lines = ["=== SeaFront TEST(2,480 張,與 train/val 場景零重疊)===", ""]
    for i, n in enumerate(names):
        lines.append(f"{n:>9}: P={r.box.p[i]:.3f}  R={r.box.r[i]:.3f}  "
                     f"mAP50={r.box.ap50[i]:.3f}  mAP50-95={r.box.ap[i]:.3f}")
    lines.append(f"{'all':>9}: mAP50={r.box.map50:.3f}  mAP50-95={r.box.map:.3f}")

    # ---- 2) 抽樣畫框(demo 素材):8 張有破損 + 4 張無破損 ----
    rng = random.Random(0)
    has_dmg, no_dmg = [], []
    for lbl in TEST_LBLS.iterdir():
        txt = lbl.read_text().strip()
        (has_dmg if any(l.split()[0] == "1" for l in txt.splitlines() if l)
         else no_dmg).append(lbl.stem)
    samples = rng.sample(has_dmg, 8) + rng.sample(no_dmg, 4)
    out_dir = run / "test_eval" / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem in samples:
        pr = model.predict(str(TEST_IMGS / f"{stem}.png"),
                           conf=0.25, imgsz=640, verbose=False)[0]
        pr.save(str(out_dir / f"{stem}_pred.jpg"))
    lines += ["", f"抽樣畫框 12 張 → {out_dir}"]

    summary = "\n".join(lines)
    print("\n" + summary)
    (run / "test_eval" / "summary.txt").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
