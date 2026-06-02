import sys
import os
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.ingestion import ingest_document_from_bytes
from app.embeddings import embed_chunks, embed_image

# Use a lithography paper from your papers.json
arxiv_id = "2506.03345v1"
url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

print(f"Fetching {url}...")
response = requests.get(url, timeout=30)
pdf_bytes = response.content

result = ingest_document_from_bytes(
    pdf_bytes=pdf_bytes,
    metadata={
        "paper_id": arxiv_id,
        "title": "Test paper",
        "category": "OPC",
        "year": "2002",
        "source": "arxiv",
    }
)

# Test text embeddings
print("\nTesting text embeddings...")
embeddings = embed_chunks(result["chunks"][:3])  # test first 3 chunks only
print(f"Text embedding shape: {len(embeddings)} chunks x {len(embeddings[0])} dims")

# Test image embeddings
print("\nTesting image embeddings...")
for img in result["images"]:
    embedding = embed_image(img)
    if embedding:
        print(f"  Page {img['page']}:")
        print(f"  CLIP embedding dims: {len(embedding['clip_embedding'])}")
        print(f"  Reasoning: {img.get('reasoning', '')[0:200]}...")
    else:
        print(f"  Page {img['page']}: embedding failed")