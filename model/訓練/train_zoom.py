# -*- coding: utf-8 -*-
"""
特寫加強版訓練(在 train.py 基礎上加入 closeup 子集 + 尺度增強調整)。

先跑 make_closeups.py 再執行:
    python train_zoom.py

與 train.py 的差異:
- 資料:data.yaml 的 train 已含 images/closeup(6,470 張特寫)
- scale 0.5 → 0.8(線上縮放範圍擴大到 0.2~1.8 倍)
- mosaic 1.0 → 0.5(mosaic 會把物件變小,偏向遠景,降低以平衡特寫方向)
- 輸出 runs\damage_yolo11s_zoom\
"""
from pathlib import Path

DATA = Path(r"D:\Arnold_chun_yen\container_damage\dataset\data.yaml")
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def main():
    import torch
    from ultralytics import YOLO

    if not (DATA.parent / "images" / "closeup").exists():
        raise SystemExit("找不到 closeup 子集,請先執行 make_closeups.py")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("警告:偵測不到 CUDA GPU,將用 CPU 訓練(會非常慢)。")

    model = YOLO(str(HERE / "yolo11s.pt"))
    model.train(
        data=str(DATA),
        epochs=60,
        patience=15,
        imgsz=640,
        batch=-1,
        device=0,
        workers=8,
        project=str(RUNS),
        name="damage_yolo11s_zoom",
        # ---- sim-to-real 增強(同 train.py)----
        hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,
        degrees=5.0, translate=0.15,
        fliplr=0.5, mixup=0.1,
        # ---- 特寫方向的調整 ----
        scale=0.8,      # 線上縮放 0.2~1.8 倍
        mosaic=0.5,     # 降低 mosaic 的「縮小偏置」
        plots=True,
    )

    from plot_metrics import make_plot
    save_dir = Path(model.trainer.save_dir)
    out = make_plot(save_dir)
    print("\n=== 完成 ===")
    print("最佳權重:", save_dir / "weights" / "best.pt")
    print("Loss+MacroF1 圖:", out)


if __name__ == "__main__":
    main()
