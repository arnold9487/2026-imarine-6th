# -*- coding: utf-8 -*-
"""
訓練 2 類貨櫃破損偵測模型(YOLO11s,RTX 5070)。

直接執行即可:
    python train.py

- 資料:D:\\Arnold_chun_yen\\container_damage\\dataset(先跑 build_dataset.py)
- 輸出:本資料夾下的 runs\\damage_yolo11s*\\
- 訓練結束自動產出 loss_macroF1.png(Loss + Macro F1 合併一張)
"""
from pathlib import Path

DATA = Path(r"D:\Arnold_chun_yen\container_damage\dataset\data.yaml")
RUNS = Path(__file__).resolve().parent / "runs"


def main():
    import torch
    from ultralytics import YOLO

    if not DATA.exists():
        raise SystemExit(f"找不到 {DATA},請先執行 build_dataset.py")
    if not torch.cuda.is_available():
        print("警告:偵測不到 CUDA GPU,將用 CPU 訓練(會非常慢)。"
              "請確認 PyTorch 是裝 cu128 版(見 訓練說明.md)。")
    else:
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolo11s.pt")   # 首次執行會自動下載 COCO 預訓練權重
    model.train(
        data=str(DATA),
        epochs=60,
        patience=15,             # 15 個 epoch 無進步就提前停
        imgsz=640,
        batch=-1,                # 依 VRAM 自動決定 batch(5070 12GB 約可到 32)
        device=0,
        workers=8,
        project=str(RUNS),
        name="damage_yolo11s",
        # ---- 縮小 sim-to-real gap 的增強(合成圖太乾淨)----
        hsv_h=0.02, hsv_s=0.8, hsv_v=0.5,   # 色彩抖動加大
        degrees=5.0, translate=0.15, scale=0.5,
        fliplr=0.5, mosaic=1.0, mixup=0.1,
        # (若有安裝 albumentations,ultralytics 會自動再加模糊/CLAHE/灰階)
        plots=True,
    )

    # 訓練結束:畫 Loss + Macro F1 合併圖
    from plot_metrics import make_plot
    save_dir = Path(model.trainer.save_dir)
    out = make_plot(save_dir)
    print("\n=== 完成 ===")
    print("最佳權重:", save_dir / "weights" / "best.pt")
    print("Loss+MacroF1 圖:", out)


if __name__ == "__main__":   # Windows 多進程必須有這個守衛
    main()
