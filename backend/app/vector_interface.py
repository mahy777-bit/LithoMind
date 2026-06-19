import chromadb
import os
from app.config import CHROMA_DIR, RETRIEVE_K

# ── Initialize ChromaDB client ─────────────────────────────

def get_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)

def get_collections():
    client = get_client()
    text_collection = client.get_or_create_collection(
        name="text_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    image_collection = client.get_or_create_collection(
        name="image_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    return text_collection, image_collection

# ── Store operations ───────────────────────────────────────

def store_chunks(chunks: list, embeddings: list, metadata: dict):
    """
    Store text chunks and their embeddings in ChromaDB.
    Each chunk gets its own entry with source metadata.
    """
    text_collection, _ = get_collections()

    ids = []
    documents = []
    metadatas = []
    embeds = []

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{metadata['paper_id']}_chunk_{i}"
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({
            "paper_id": metadata["paper_id"],
            "title": metadata["title"],
            "category": metadata["category"],
            "year": metadata["year"],
            "source": metadata["source"],
            "type": "text"
        })
        embeds.append(embedding)

    text_collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeds
    )
    print(f"  Stored {len(chunks)} text chunks")

def store_image(image: dict, embedding: dict, metadata: dict):
    """
    Store image CLIP embedding + reasoning in ChromaDB.
    """
    _, image_collection = get_collections()

    image_id = f"{metadata['paper_id']}_image_p{image['page']}_{image['index']}"

    image_collection.upsert(
        ids=[image_id],
        documents=[image.get("reasoning", "")],
        metadatas=[{
            "paper_id": metadata["paper_id"],
            "title": metadata["title"],
            "category": metadata["category"],
            "year": metadata["year"],
            "source": metadata["source"],
            "page": image["page"],
            "type": "image"
        }],
        embeddings=[embedding["clip_embedding"]]
    )

# ── Search operations ──────────────────────────────────────

def search_text(query_embedding: list, k: int = RETRIEVE_K) -> list:
    """
    Search text chunks collection.
    Returns list of dicts with chunk text and metadata.
    """
    text_collection, _ = get_collections()
    results = text_collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    return _format_results(results)

def search_images(query_embedding: list, k: int = RETRIEVE_K) -> list:
    """
    Search image chunks collection.
    Returns list of dicts with reasoning text and metadata.
    """
    _, image_collection = get_collections()
    results = image_collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    return _format_results(results)

def search_all(text_query_embedding: list, clip_query_embedding: list, k: int = RETRIEVE_K) -> list:
    """
    Search both collections and combine results.
    text_query_embedding: question embedded with text_model (768-dim)
    clip_query_embedding: question embedded with CLIP text encoder (512-dim)
    """
    text_results = search_text(text_query_embedding, k)
    image_results = search_images(clip_query_embedding, k)

    all_results = text_results + image_results
    all_results.sort(key=lambda x: x["distance"])

    return all_results[:k]

# ── Helper ─────────────────────────────────────────────────

def _format_results(results: dict) -> list:
    formatted = []
    if not results["documents"] or not results["documents"][0]:
        return formatted
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        formatted.append({
            "content": doc,
            "metadata": meta,
            "distance": dist
        })
    return formatted