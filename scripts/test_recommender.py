import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.recommender import recommend

# Use a layout-style image — for testing, pull one from a paper we've already ingested
# (any image already in image_chunks works, since we're testing recommend() logic)
import requests

# Re-fetch the same test paper's images, grab one to use as a "user-provided layout"
from app.ingestion import ingest_document_from_bytes

arxiv_id = "2506.03345v1"
url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
response = requests.get(url, timeout=30)
pdf_bytes = response.content

result = ingest_document_from_bytes(
    pdf_bytes=pdf_bytes,
    metadata={"paper_id": arxiv_id, "title": "Test", "category": "test", "year": "2023", "source": "arxiv"}
)

test_image_bytes = result["images"][0]["bytes"]

# Test without description
print("=== Without description ===")
output = recommend(test_image_bytes)
print(output["recommendation"])
print(f"\nSources: {len(output['sources'])}")

print("\n\n=== With description ===")
output = recommend(test_image_bytes, description="Dense SEM defect pattern, corner-to-corner spacing critical")
print(output["recommendation"])
print(f"\nSources: {len(output['sources'])}")