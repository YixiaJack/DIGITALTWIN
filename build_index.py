"""
构建分时期的向量索引，使用 LlamaIndex + ChromaDB。
"""

from pathlib import Path
from typing import List

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import PERIODS, require_api_key

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"


def gather_files(folders: List[Path]) -> List[Path]:
    """收集所有支持的文件。"""

    exts = {".docx", ".txt", ".md"}
    files: List[Path] = []
    for folder in folders:
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in exts:
                files.append(path)
    return files


def build_period_index(period_folder: Path) -> None:
    """
    构建单个时期的索引（含通用资料）。

    Args:
        period_folder: data 下对应时期的文件夹。
    """

    require_api_key()
    general_folder = DATA_DIR / "通用"
    files = gather_files([period_folder, general_folder])
    if not files:
        print(f"[build_period_index] 未找到可处理文件: {period_folder}")
        return

    print(f"[build_period_index] 读取 {len(files)} 个文件...")
    reader = SimpleDirectoryReader(input_files=[str(f) for f in files])
    documents = reader.load_data()

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # 使用 config 中定义的英文 collection_name，避免中文命名报错
    try:
        collection_name = next(
            period.collection_name
            for period in PERIODS.values()
            if period.folder == period_folder.name
        )
    except StopIteration as exc:  # noqa: PERF203
        raise RuntimeError(f"未在 PERIODS 中找到对应配置: {period_folder.name}") from exc
    # 重建时清理旧集合，避免重复
    existing = {col.name for col in client.list_collections()}
    if collection_name in existing:
        client.delete_collection(name=collection_name)

    collection = client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    print(f"[build_period_index] 完成索引: {collection_name}")


def main() -> None:
    """遍历所有时期并构建索引。"""

    for key, period in PERIODS.items():
        folder = DATA_DIR / period.folder
        print(f"\n=== 构建时期 {period.name} ({key}) ===")
        build_period_index(folder)
    print("\n全部索引构建完成。")


if __name__ == "__main__":
    main()
