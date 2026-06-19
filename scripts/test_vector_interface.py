import sys
import os
import requests

# Allows importing from backend/app/ even though this script lives in scripts/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.ingestion import ingest_document_from_bytes
from app.embeddings import embed_chunks, embed_image, text_model, embed_text_clip
from app.vector_interface import store_chunks, store_image, search_text, search_images

# ── Step 1: Get a real paper to work with ──────────────────
arxiv_id = "2506.03345v1"
url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

print(f"Fetching {url}...")
response = requests.get(url, timeout=30)
pdf_bytes = response.content

metadata = {
    "paper_id": arxiv_id,
    "title": "Test paper",
    "category": "Defect Inspection",
    "year": "2023",
    "source": "arxiv",
}

# ── Step 2: Run the ingestion pipeline ─────────────────────
result = ingest_document_from_bytes(pdf_bytes=pdf_bytes, metadata=metadata)

# ── Step 3: Embed and store TEXT chunks ────────────────────
print("\nEmbedding and storing text chunks...")
embeddings = embed_chunks(result["chunks"])
store_chunks(result["chunks"], embeddings, metadata)

# ── Step 4: Embed and store IMAGES ─────────────────────────
print("\nEmbedding and storing images...")
for img in result["images"]:
    embedding = embed_image(img)
    if embedding:
        store_image(img, embedding, metadata)

print("\nStorage complete. Testing search...")

# ── Step 5: Embed the SAME question two different ways ─────
question = "What method is used for defect classification?"

text_query_embedding = text_model.encode([question])[0].tolist()
clip_query_embedding = embed_text_clip(question)

# ── Step 6: Search TEXT collection ─────────────────────────
text_results = search_text(text_query_embedding, k=3)
print(f"\nText search results for: '{question}'")
for r in text_results:
    print(f"  Distance: {r['distance']:.3f} | {r['content'][:100]}...")

# ── Step 7: Search IMAGE collection ────────────────────────
image_results = search_images(clip_query_embedding, k=2)
print(f"\nImage search results for: '{question}'")
for r in image_results:
    print(f"  Distance: {r['distance']:.3f} | Page {r['metadata']['page']} | {r['content'][:100]}...")