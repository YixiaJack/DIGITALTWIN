"""
使用 GPT-4o Vision 生成作品描述。
"""

import base64
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from config import require_api_key


def encode_image(image_path: Path) -> str:
    """将图片转为 base64 字符串。"""

    with open(image_path, "rb") as fp:
        return base64.b64encode(fp.read()).decode("utf-8")


def describe_artwork(image_path: Path, period: str) -> str:
    """
    使用 GPT-4o Vision 生成单张图片的详细艺术描述。

    Args:
        image_path: 图片路径。
        period: 时期标识，用于提示模型代入该阶段视角。

    Returns:
        描述文本。
    """

    client = OpenAI(api_key=require_api_key())
    base64_image = encode_image(image_path)

    messages = [
        {
            "role": "system",
            "content": f"你是绘本艺术家“贵图子”在 {period} 时期的视角，请用第一人称进行专业、细腻的作品描述。",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "请描述这张图片，覆盖画面内容、构图、色彩、技法和情感氛围，并结合该时期的创作特征。",
                },
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ],
        },
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.4,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"调用 OpenAI 接口失败: {exc}") from exc

    content = response.choices[0].message.content
    return content if isinstance(content, str) else "".join(part["text"] for part in content or [])


def iter_images(folder: Path) -> Iterable[Path]:
    """遍历文件夹内所有图片文件。"""

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def batch_describe_images(image_folder: Path, period: str, output_file: Path) -> None:
    """
    批量处理文件夹内所有图片，输出描述到 txt。

    Args:
        image_folder: 图片所在目录。
        period: 时期标识。
        output_file: 结果保存路径。
    """

    if not image_folder.exists():
        print(f"[batch_describe_images] 目录不存在: {image_folder}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fp:
        for image_path in iter_images(image_folder):
            print(f"[describe] 处理 {image_path}")
            try:
                description = describe_artwork(image_path, period)
            except Exception as exc:  # noqa: BLE001
                print(f"[describe] 生成描述失败 {image_path}: {exc}")
                continue
            fp.write(f"## {image_path.name}\n{description}\n\n")
    print(f"[batch_describe_images] 生成完成 -> {output_file}")


if __name__ == "__main__":
    # 示例：python describe_images.py data/2016-2017_早期探索/images 2016-2017 output/2016-2017_images.txt
    import sys

    if len(sys.argv) != 4:
        print("用法: python describe_images.py <image_folder> <period> <output_file>")
    else:
        batch_describe_images(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
