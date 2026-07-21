"""檢視 .lib 檔案結構的探索性腳本。

用途：讓你直觀理解 Liberty 檔案裡有什麼、我們要抽哪些欄位當訓練資料。

執行：
    conda activate nldm
    python src/inspect_lib.py

會做什麼：
    1. 讀入 ASAP7 INVBUF TT corner .lib（約 300 KB，秒解）
    2. 列出前 10 個 cell 名稱
    3. 挑一個代表 cell（INVx1），印出 pin 清單
    4. 印出該 pin 的 cell_rise LUT：
         - index_1（input slew 陣列）
         - index_2（output load 陣列）
         - values（delay 矩陣）
    這個 (index_1, index_2, values) 就是我們要「攤平」成 (slew, load, delay)
    三元組的訓練資料。
"""

from __future__ import annotations

from pathlib import Path

from liberty.parser import parse_liberty


LIB_PATH = Path("data/external/asap7/asap7sc7p5t_INVBUF_RVT_TT_nldm_220122.lib")


def hr(title: str = "") -> None:
    line = "=" * 60
    if title:
        print(f"\n{line}\n {title}\n{line}")
    else:
        print(line)


def main() -> None:
    hr(f"讀取 {LIB_PATH.name}")
    text = LIB_PATH.read_text(encoding="utf-8")
    library = parse_liberty(text)
    print(f"檔案大小       : {LIB_PATH.stat().st_size:,} bytes")
    print(f"Library 名稱   : {library.args[0]}")

    # -------------------------------------------------------
    # Step 1: 列出所有 cell
    # -------------------------------------------------------
    cells = [g for g in library.get_groups("cell")]
    hr(f"Step 1: cell 總數 = {len(cells)}（顯示前 10 個）")
    for c in cells[:10]:
        print(f"  - {c.args[0]}")

    # -------------------------------------------------------
    # Step 2: 挑一個代表 cell（找 INV 家族的第一個）
    # -------------------------------------------------------
    target = next((c for c in cells if "INV" in c.args[0]), cells[0])
    cell_name = target.args[0]
    hr(f"Step 2: 檢視 cell = {cell_name}")

    pins = list(target.get_groups("pin"))
    print(f"pin 數量       : {len(pins)}")
    for p in pins:
        pin_name = p.args[0]
        direction = p.get_attribute("direction")
        print(f"  - {pin_name}  (direction={direction})")

    # -------------------------------------------------------
    # Step 3: 挑輸出 pin，抽 cell_rise LUT
    # -------------------------------------------------------
    output_pin = next(
        (p for p in pins if str(p.get_attribute("direction")).strip('"') == "output"),
        None,
    )
    if output_pin is None:
        print("找不到 output pin，結束。")
        return

    hr(f"Step 3: 檢視 output pin = {output_pin.args[0]}")

    timings = list(output_pin.get_groups("timing"))
    print(f"timing arc 數量: {len(timings)}")

    if not timings:
        return

    # 挑第一個 timing arc
    t = timings[0]
    related_pin = t.get_attribute("related_pin")
    timing_sense = t.get_attribute("timing_sense")
    print(f"\n代表 timing arc:")
    print(f"  related_pin  : {related_pin}")
    print(f"  timing_sense : {timing_sense}")

    # 抽 cell_rise LUT
    rise = next(iter(t.get_groups("cell_rise")), None)
    if rise is None:
        print("  (無 cell_rise)")
        return

    template_name = rise.args[0] if rise.args else "(no template)"
    print(f"\ncell_rise 使用範本: {template_name}")

    # index_1 = input transition (slew) 掃描點
    # index_2 = output capacitance (load) 掃描點
    # values  = 2D delay 矩陣
    idx1 = rise.get_array("index_1")
    idx2 = rise.get_array("index_2")
    vals = rise.get_array("values")

    print(f"\n  index_1 (input slew)   shape={idx1.shape}:")
    print(f"    {idx1}")
    print(f"\n  index_2 (output load)  shape={idx2.shape}:")
    print(f"    {idx2}")
    print(f"\n  values (delay matrix)  shape={vals.shape}:")
    for i, row in enumerate(vals):
        print(f"    slew[{i}]={idx1.flatten()[i]:>8.3f}:  " + "  ".join(f"{v:7.3f}" for v in row))

    # -------------------------------------------------------
    # Step 4: 展平成訓練資料格式
    # -------------------------------------------------------
    hr("Step 4: 展平成 (slew, load, delay) 三元組")
    slew_flat = idx1.flatten()
    load_flat = idx2.flatten()

    triples = []
    for i, s in enumerate(slew_flat):
        for j, l in enumerate(load_flat):
            triples.append((s, l, vals[i, j]))
    print(f"這個 timing arc 的訓練樣本數: {len(triples)}")
    print(f"前 5 筆:")
    for s, l, d in triples[:5]:
        print(f"  slew={s:>8.3f}  load={l:>10.4f}  delay={d:>10.4f}")

    hr("完成")
    print("這就是資料 pipeline 的核心：把每個 cell 的 cell_rise / cell_fall / rise_transition")
    print("/ fall_transition 都攤平成三元組，就是 ANN 的訓練資料了。")


if __name__ == "__main__":
    main()
