#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重組並解壓 SeaFront 分割 zip（SeaFront_v1_0_0.zip.001 ... .005）。

直接執行即可，不需要任何參數：
    python reconstruct_and_extract_seafront.py

流程：
  1. 依序串接 .001 ~ .005 成一個完整 zip（暫存在輸出磁碟 D:，不佔用 C:）
  2. 檢查 zip 有效性後解壓到 OUTPUT_DIR
  3. 解壓過程 zipfile 會逐檔驗 CRC，資料損毀會直接報錯
  4. 完成後刪除合併暫存檔
只處理 PREFIX 指定的分割檔；SeaFront_v1_0_0_TEST.zip 不會被動到。
"""
import re
import sys
import zipfile
from pathlib import Path

# ===== 設定（已寫死，直接執行即可）=====
PARTS_DIR = Path(__file__).resolve().parent          # 本腳本所在資料夾（辨識模型）
PREFIX = "SeaFront_v1_0_0"                            # 只抓這個前綴的 .zip.NNN
OUTPUT_DIR = Path(r"D:\Arnold_chun_yen")             # 解壓目的地
BUFFER_SIZE = 16 * 1024 * 1024                        # 合併時的緩衝區（16 MB）
# =======================================

GB = 1024 ** 3


def find_parts(parts_dir: Path, prefix: str):
    """找出 <prefix>.zip.001, .002 ... 並依編號排序。"""
    regex = re.compile(re.escape(prefix) + r"\.zip\.(\d+)$")
    parts = []
    for p in parts_dir.iterdir():
        m = regex.match(p.name)
        if m:
            parts.append((int(m.group(1)), p))
    parts.sort(key=lambda t: t[0])
    return [p for _, p in parts]


def combine_parts(parts, out_path: Path, buffer_size: int):
    total = sum(p.stat().st_size for p in parts)
    written = 0
    with out_path.open("wb") as w:
        for i, p in enumerate(parts, start=1):
            print(f"  合併部件 {i}/{len(parts)}: {p.name} ({p.stat().st_size / GB:.2f} GB)")
            with p.open("rb") as r:
                while True:
                    chunk = r.read(buffer_size)
                    if not chunk:
                        break
                    w.write(chunk)
                    written += len(chunk)
            print(f"    進度 {written / GB:.2f} / {total / GB:.2f} GB")


def main():
    if not PARTS_DIR.exists():
        print("找不到來源資料夾：", PARTS_DIR)
        sys.exit(1)

    parts = find_parts(PARTS_DIR, PREFIX)
    if not parts:
        print(f"在 {PARTS_DIR} 找不到任何 {PREFIX}.zip.* 分割檔")
        sys.exit(1)

    print(f"找到 {len(parts)} 個分割檔：")
    for p in parts:
        print("   ", p.name)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 合併暫存檔放在輸出磁碟（D:），避免佔用 C:；完成後刪除
    combined_zip = OUTPUT_DIR / f"_{PREFIX}_combined_tmp.zip"
    try:
        print("\n正在合併到暫存檔：", combined_zip)
        combine_parts(parts, combined_zip, BUFFER_SIZE)

        print("合併完成，檢查 zip 有效性 ...")
        if not zipfile.is_zipfile(combined_zip):
            print("錯誤：合併後不是有效 zip，請確認分割檔完整且順序正確（.001~.005）。")
            sys.exit(2)
        print("zip 表頭檢查通過。")

        print("\n開始解壓到：", OUTPUT_DIR)
        with zipfile.ZipFile(combined_zip, "r") as z:
            members = z.infolist()
            n = len(members)
            for idx, m in enumerate(members, start=1):
                z.extract(m, OUTPUT_DIR)   # extract 會驗 CRC，損毀會丟 BadZipFile
                if idx % 2000 == 0 or idx == n:
                    print(f"    已解壓 {idx}/{n} 個項目")
        print("解壓完成。")

    except zipfile.BadZipFile as e:
        print("錯誤：解壓時發生 BadZipFile（檔案可能損毀或順序錯誤）：", e)
        sys.exit(3)
    finally:
        # 清掉合併暫存檔
        if combined_zip.exists():
            try:
                combined_zip.unlink()
                print("已刪除合併暫存檔：", combined_zip.name)
            except OSError as e:
                print("提醒：暫存檔刪除失敗，可手動刪除：", combined_zip, e)


if __name__ == "__main__":
    main()
