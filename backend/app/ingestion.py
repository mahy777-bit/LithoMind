import fitz  # pymupdf
import io
import base64
from PIL import Image
from anthropic import Anthropic
from app.config import (
    ANTHROPIC_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP)

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Text extraction ────────────────────────────────────────

def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# ── Chunking ───────────────────────────────────────────────

def chunk_text(text: str) -> list:
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    overlap_buffer = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= CHUNK_SIZE:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                overlap_buffer = current_chunk[-CHUNK_OVERLAP:]
            current_chunk = overlap_buffer + " " + sentence
            overlap_buffer = ""

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# ── Image extraction ───────────────────────────────────────

def extract_images(pdf_bytes: bytes) -> list:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    images = []

    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes))

            if pil_image.width < 100 or pil_image.height < 100:
                continue

            # Extract text from page before, current page, and page after
            start_page = max(0, page_num - 1)
            end_page = min(total_pages - 1, page_num + 1)
            
            context = ""
            for ctx_page_num in range(start_page, end_page + 1):
                context += doc[ctx_page_num].get_text()

            images.append({
                "pil_image": pil_image,
                "bytes": image_bytes,
                "page": page_num + 1,
                "index": img_index,
                "image_type": None,
                "context": context[:2000],  # cap at 2000 chars
            })
    return images

# ── Main entry point ───────────────────────────────────────

def ingest_document_from_bytes(pdf_bytes: bytes, metadata: dict) -> dict:
    """
    Takes PDF bytes and metadata.
    Returns text chunks and classified images — nothing written to disk.
    """
    # Extract and chunk text
    text = extract_text(pdf_bytes)
    chunks = chunk_text(text)

    # Extract images
    images = extract_images(pdf_bytes)

    print(f"  → {len(chunks)} text chunks, {len(images)} images")

    return {
        "chunks": chunks,
        "images": images,
        "metadata": metadata
    }