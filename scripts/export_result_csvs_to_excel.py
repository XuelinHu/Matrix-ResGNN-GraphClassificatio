"""把实验结果 CSV 汇总文件批量转换为可打开的 Excel xlsx 文件。"""
from __future__ import annotations

import csv
import html
import zipfile
from pathlib import Path


# 仓库根目录：用于把脚本中的相对路径统一定位到项目根路径。
ROOT = Path(__file__).resolve().parents[1]


def col_name(index: int) -> str:
    """把从零开始的列下标转换为 Excel 列名。"""
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def cell_xml(value: str, row_idx: int, col_idx: int) -> str:
    """把单元格文本转换为 xlsx 工作表 XML 片段。"""
    ref = f"{col_name(col_idx)}{row_idx}"
    text = value.strip()
    if text:
        try:
            float(text.replace(",", ""))
            is_number = all(ch not in text for ch in [" ", "_"]) and not text.startswith("0")
        except ValueError:
            is_number = False
        if is_number:
            return f'<c r="{ref}"><v>{html.escape(text)}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{html.escape(value)}</t></is></c>'


def write_xlsx(rows: list[list[str]], target: Path) -> None:
    """用标准库写出单工作表 xlsx 文件。"""
    sheet_rows = []
    for row_idx, row in enumerate(rows, start=1):
        cells = "".join(cell_xml(value, row_idx, col_idx) for col_idx, value in enumerate(row))
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheetData>'
        + "".join(sheet_rows)
        + '</sheetData></worksheet>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as xlsx:
        xlsx.writestr("[Content_Types].xml", content_types)
        xlsx.writestr("_rels/.rels", rels)
        xlsx.writestr("xl/workbook.xml", workbook)
        xlsx.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        xlsx.writestr("xl/worksheets/sheet1.xml", worksheet)


def convert_dir(source_dir: Path, target_dir: Path) -> list[Path]:
    """把目录下的所有 CSV 文件逐个转换成 xlsx 文件。"""
    written: list[Path] = []
    for source in sorted(source_dir.glob("*.csv")):
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.reader(handle))
        target = target_dir / f"{source.stem}.xlsx"
        write_xlsx(rows, target)
        written.append(target)
    return written


def main() -> None:
    """脚本主入口，串联参数解析、数据读取、处理和结果写出。"""
    outputs = []
    # 主实验汇总 CSV：从 records/LATEST/summaries 导出到 records/LATEST/excel/summaries。
    outputs.extend(convert_dir(ROOT / "records/LATEST/summaries", ROOT / "records/LATEST/excel/summaries"))
    # 机制分析汇总 CSV：从 records/LATEST/mechanism_summaries 导出到对应 Excel 目录。
    outputs.extend(
        convert_dir(
            ROOT / "records/LATEST/mechanism_summaries",
            ROOT / "records/LATEST/excel/mechanism_summaries",
        )
    )
    for path in outputs:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
