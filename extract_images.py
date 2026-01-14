"""
从 Word 文档批量提取图片。
"""

import os
from pathlib import Path
from typing import Iterable

from docx import Document


DATA_DIR = Path(__file__).parent / "data"


def extract_images_from_docx(docx_path: Path, output_folder: Path) -> int:
    """
    从单个 docx 提取所有嵌入图片。

    Args:
        docx_path: Word 文档路径。
        output_folder: 输出图片文件夹。

    Returns:
        成功保存的图片数量。
    """

    document = Document(docx_path)
    output_folder.mkdir(parents=True, exist_ok=True)

    count = 0
    for part in document.part.related_parts.values():
        if not part.content_type.startswith("image/"):
            continue
        ext = os.path.splitext(str(part.partname))[-1] or ".png"
        count += 1
        filename = output_folder / f"image_{count:03d}{ext}"
        with open(filename, "wb") as fp:
            fp.write(part.blob)
    print(f"[extract_images_from_docx] {docx_path.name}: 提取 {count} 张图片 -> {output_folder}")
    return count


def iter_docx_files(root: Path) -> Iterable[Path]:
    """遍历目录下的所有 docx 文件。"""

    return sorted(root.rglob("*.docx"))


def batch_extract_all() -> None:
    """遍历 data/ 下所有 docx，提取图片到同目录 images/ 子文件夹。"""

    if not DATA_DIR.exists():
        print(f"[batch_extract_all] 数据目录不存在: {DATA_DIR}")
        return

    total = 0
    for docx_path in iter_docx_files(DATA_DIR):
        output_folder = docx_path.parent / "images"
        try:
            total += extract_images_from_docx(docx_path, output_folder)
        except Exception as exc:  # noqa: BLE001
            print(f"[batch_extract_all] 处理 {docx_path} 失败: {exc}")
    print(f"[batch_extract_all] 处理完成，总计保存 {total} 张图片")


if __name__ == "__main__":
    batch_extract_all()
