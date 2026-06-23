import json
import os
import sys
import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.ingestion import ingest_document_from_bytes
from app.embeddings import embed_chunks, embed_image
from app.vector_interface import store_chunks, store_image
from app.storage import upload_db
from app.config import PAPERS_FILE

def load_papers() -> list:
    with open(PAPERS_FILE, "r") as f:
        return json.load(f)

def fetch_pdf_bytes(paper: dict) -> bytes | None:
    try:
        url = paper.get("url")
        if not url:
            print(f"  No URL stored for paper {paper.get('id')} — skipping")
            return None

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"  Fetch failed: {e}")
        return None

def build():
    papers = load_papers()
    print(f"Building knowledge base from {len(papers)} papers...\n")

    success = 0
    failed = 0

    for i, paper in enumerate(papers):
        print(f"[{i+1}/{len(papers)}] {paper['title'][:60]}...")

        pdf_bytes = fetch_pdf_bytes(paper)
        if not pdf_bytes:
            print("  Skipped — fetch failed")
            failed += 1
            continue

        metadata = {
            "paper_id": paper["id"],
            "title": paper["title"],
            "category": paper["category"],
            "year": str(paper.get("year", "unknown")),
            "source": paper["source"],
        }

        try:
            result = ingest_document_from_bytes(pdf_bytes, metadata)
        except Exception as e:
            print(f"  Skipped — ingestion failed: {e}")
            failed += 1
            continue

        try:
            embeddings = embed_chunks(result["chunks"])
            store_chunks(result["chunks"], embeddings, metadata)
        except Exception as e:
            print(f"  Text embedding/storage failed: {e}")

        for img in result["images"]:
            try:
                embedding = embed_image(img)
                if embedding:
                    store_image(img, embedding, metadata)
            except Exception as e:
                print(f"  Image embedding/storage failed: {e}")

        success += 1
        print(f"  Done — {len(result['chunks'])} chunks, {len(result['images'])} images")

    print(f"\nBuild complete — {success} succeeded, {failed} failed")

    upload = input("\nUpload knowledge base to Hugging Face? (y/n): ").strip().lower()
    if upload == "y":
        upload_db()
    else:
        print("Skipped upload. Run again or call storage.upload_db() manually when ready.")

if __name__ == "__main__":
    build()