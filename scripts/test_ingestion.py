import sys
import os
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.ingestion import ingest_document_from_bytes

# Test with one arXiv paper
arxiv_id = "2506.03345v1"
url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

print(f"Fetching {url}...")
response = requests.get(url, timeout=30)
pdf_bytes = response.content
print(f"Downloaded {len(pdf_bytes)} bytes")

result = ingest_document_from_bytes(
    pdf_bytes=pdf_bytes,
    metadata={
        "paper_id": arxiv_id,
        "title": "Test paper",
        "category": "ML for Lithography",
        "year": "2023",
        "source": "arxiv",
    }
)

print(f"\nResults:")
print(f"Text chunks: {len(result['chunks'])}")
print(f"Images: {len(result['images'])}")
print(f"\nFirst chunk preview:")
print(result['chunks'][0][:300])