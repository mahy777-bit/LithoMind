import os
from anthropic import Anthropic
from app.config import ANTHROPIC_API_KEY, LLM_MODEL
from app.embeddings import text_model, embed_text_clip
from app.vector_interface import search_all

client = Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are LithoMind, a semiconductor lithography and OPC/EDA expert assistant.

Answer questions using ONLY the provided sources below. Each source is labeled with a number, paper title, and page number.

Rules:
- Cite sources using their label, e.g. "According to [Source 1]..."
- If the sources do not contain enough information to answer the question, say so clearly rather than guessing
- Be technically precise and concise
- If multiple sources support the same point, cite all of them
"""

def answer_question(question: str) -> dict:
    """
    Full pipeline: embed question -> search both collections ->
    format sources -> ask Claude -> return answer + source list
    """
    # Step 1: embed question for both vector spaces
    text_query_embedding = text_model.encode([question])[0].tolist()
    clip_query_embedding = embed_text_clip(question)

    # Step 2: search both collections, combined and ranked
    results = search_all(text_query_embedding, clip_query_embedding)

    if not results:
        return {
            "answer": "I don't have any information in my knowledge base relevant to this question.",
            "sources": []
        }

    # Step 3: format sources for Claude with explicit labels
    sources_text, sources_list = _format_sources(results)

    # Step 4: ask Claude
    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Sources:\n{sources_text}\n\nQuestion: {question}"
        }]
    )

    return {
        "answer": message.content[0].text,
        "sources": sources_list
    }

def _format_sources(results: list) -> tuple[str, list]:
    sources_text = ""
    sources_list = []

    for i, result in enumerate(results):
        label = f"Source {i+1}"
        meta = result["metadata"]
        title = meta.get("title", "Unknown")
        source_type = meta.get("type")

        if source_type == "image":
            page = meta.get("page", "N/A")
            sources_text += f"[{label}: \"{title}\", page {page}]\n{result['content']}\n\n"
        else:
            sources_text += f"[{label}: \"{title}\"]\n{result['content']}\n\n"

        sources_list.append({
            "label": label,
            "title": title,
            "page": meta.get("page") if source_type == "image" else None,
            "paper_id": meta.get("paper_id"),
            "type": source_type,
        })

    return sources_text, sources_list