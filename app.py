"""
Streamlit 前端：与不同创作时期的“贵图子”对话。
"""

from pathlib import Path
from typing import List, Tuple

import chromadb
import streamlit as st
from openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import PERIODS, require_api_key

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"
PERSONA_DIR = BASE_DIR / "personas"


@st.cache_resource(show_spinner=False)
def load_index(period_key: str) -> VectorStoreIndex:
    """从 ChromaDB 加载已有索引。"""

    period = PERIODS[period_key]
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(period.collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)


@st.cache_data(show_spinner=False)
def load_persona(period_key: str) -> str:
    """读取对应时期的人格提示词文件。"""

    period = PERIODS[period_key]
    path = PERSONA_DIR / period.persona_file
    if not path.exists():
        raise FileNotFoundError(f"未找到人格文件: {path}")
    return path.read_text(encoding="utf-8")


def _format_context(nodes) -> str:
    """将检索节点拼接为上下文字符串。"""

    texts: List[str] = []
    for node in nodes:
        if hasattr(node, "node"):
            texts.append(node.node.get_content())
        elif hasattr(node, "get_content"):
            texts.append(node.get_content())
        else:
            texts.append(str(node))
    return "\n\n".join(texts)


def chat_with_artist(
    period_key: str,
    user_question: str,
    chat_history: List[dict],
) -> Tuple[str, List[dict]]:
    """
    检索相关文档并与 GPT-4o 对话。

    Args:
        period_key: 选中的时期。
        user_question: 用户问题。
        chat_history: 现有对话历史。

    Returns:
        模型回复文本与更新后的历史。
    """

    client = OpenAI(api_key=require_api_key())
    persona = load_persona(period_key)
    index = load_index(period_key)
    retriever = index.as_retriever(similarity_top_k=5)
    retrieved = retriever.retrieve(user_question)
    context_text = _format_context(retrieved)

    system_prompt = (
        f"{persona}\n\n"
        "以下是与问题相关的资料片段，请以第一人称、中文回答：\n"
        f"{context_text}"
    )

    history = chat_history[-12:]  # 仅保留最近 6 轮（12 条消息）
    messages = [{"role": "system", "content": system_prompt}, *history, {"role": "user", "content": user_question}]

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.6,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"调用 OpenAI 接口失败: {exc}") from exc

    reply = completion.choices[0].message.content or ""
    updated_history = [*history, {"role": "user", "content": user_question}, {"role": "assistant", "content": reply}]
    return reply, updated_history[-12:]


def render_chat() -> None:
    """构建 Streamlit UI。"""

    st.set_page_config(page_title="贵图子 · 数字分身", page_icon="🎨")
    st.title("贵图子 · 数字分身对话")
    st.caption("选择不同创作时期，与那时的“我”对话。")

    period_keys = list(PERIODS.keys())
    default_period = period_keys[0]
    current_period = st.sidebar.radio(
        "选择时期",
        period_keys,
        index=period_keys.index(st.session_state.get("period", default_period)),
        format_func=lambda k: PERIODS[k].name,
    )

    if "period" not in st.session_state or current_period != st.session_state.get("period"):
        st.session_state["chat_history"] = []
    st.session_state["period"] = current_period

    if st.sidebar.button("清空对话"):
        st.session_state["chat_history"] = []

    chat_history: List[dict] = st.session_state.get("chat_history", [])
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("向艺术家提问..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        try:
            reply, new_history = chat_with_artist(current_period, prompt, chat_history)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return
        st.session_state["chat_history"] = new_history
        with st.chat_message("assistant"):
            st.markdown(reply)


if __name__ == "__main__":
    render_chat()
