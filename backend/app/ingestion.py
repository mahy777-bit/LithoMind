import fitz  # pymupdf
import io
import base64
from PIL import Image
from anthropic import Anthropic
from app.config import (
    ANTHROPIC_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP,
    CLIP_IMAGE_TYPES, CAPTION_IMAGE_TYPES,
    AUTO_CLASSIFY_IMAGES, CLASSIFICATION_MODEL
)

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
    images = []
    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_image = Image.open(io.BytesIO(image_bytes))

            # Skip very small images — likely icons or artifacts
            if pil_image.width < 100 or pil_image.height < 100:
                continue

            images.append({
                "pil_image": pil_image,
                "bytes": image_bytes,
                "page": page_num + 1,
                "index": img_index,
                "image_type": None,
            })
    return images

# ── Image classification ───────────────────────────────────
def classify_image(image_bytes: bytes, user_tag: str = None) -> str:
    if user_tag and user_tag.lower() in CLIP_IMAGE_TYPES + CAPTION_IMAGE_TYPES:
        return user_tag.lower()

    if not AUTO_CLASSIFY_IMAGES:
        return "diagram"

    # Detect actual image format
    pil_image = Image.open(io.BytesIO(image_bytes))
    fmt = pil_image.format.lower() if pil_image.format else "png"
    media_type_map = {
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    media_type = media_type_map.get(fmt, "image/png")

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    try:
        response = client.messages.create(
            model=CLASSIFICATION_MODEL,
            max_tokens=20,
            system=f"""Classify this semiconductor image as exactly one of: 
            {CLIP_IMAGE_TYPES + CAPTION_IMAGE_TYPES}.
            Respond with the single word only.""",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64
                    }},
                    {"type": "text", "text": "Classify this image."}
                ]
            }]
        )
        label = response.content[0].text.strip().lower()
        return label if label in CLIP_IMAGE_TYPES + CAPTION_IMAGE_TYPES else "diagram"
    except Exception as e:
        print(f"Classification failed: {e}")
        return "diagram"

# ── Main entry point ───────────────────────────────────────

def ingest_document_from_bytes(pdf_bytes: bytes, metadata: dict) -> dict:
    """
    Takes PDF bytes and metadata.
    Returns text chunks and classified images — nothing written to disk.
    """
    # Extract and chunk text
    text = extract_text(pdf_bytes)
    chunks = chunk_text(text)

    # Extract and classify images
    images = extract_images(pdf_bytes)
    for img in images:
        img["image_type"] = classify_image(img["bytes"])

    print(f"  → {len(chunks)} text chunks, {len(images)} images")

    return {
        "chunks": chunks,
        "images": images,
        "metadata": metadata
    }