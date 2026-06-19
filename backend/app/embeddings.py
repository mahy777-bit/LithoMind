import base64
import io
import torch
import numpy as np
from PIL import Image
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from app.config import (
    ANTHROPIC_API_KEY,
    TEXT_EMBEDDING_MODEL,
    IMAGE_EMBEDDING_MODEL,
    IMAGE_REASONING_MODEL
)

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# ── Load models once at module level ──────────────────────
# Loading inside functions would reload on every call — expensive

print("Loading text embedding model...")
text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)

print("Loading CLIP model...")
clip_model = CLIPModel.from_pretrained(IMAGE_EMBEDDING_MODEL)
clip_processor = CLIPProcessor.from_pretrained(IMAGE_EMBEDDING_MODEL)

# ── Text embedding ─────────────────────────────────────────

def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """
    Embed a list of text chunks.
    Returns list of embedding vectors.
    """
    embeddings = text_model.encode(chunks, show_progress_bar=True)
    return embeddings.tolist()

# ── Image reasoning generation────────────────────────────────────────
def generate_reasoning(image_bytes: bytes, context: str = "") -> str:
    """
    Claude analyzes image + surrounding paper text
    and generates rich technical interpretation.
    """
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
    context_text = f"\n\nSurrounding paper text:\n{context}" if context else ""

    try:
        response = client.messages.create(
            model=IMAGE_REASONING_MODEL,
            max_tokens=400,
            system="""You are a semiconductor lithography expert analyzing 
            a figure from a technical paper. Using both the image and the 
            surrounding paper text, provide a rich technical interpretation:
            - What is shown in the image
            - Its technical significance in lithography/OPC context
            - How it relates to the surrounding paper content
            - Key observations an engineer would make
            Respond in plain text without markdown formatting.""",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64
                    }},
                    {"type": "text", "text": 
                     f"Analyze this figure from a lithography paper.{context_text}"}
                ]
            }]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"Reasoning generation failed: {e}")
        return context  # fallback to raw context if Claude fails

# ── CLIP embedding (for layout/wafer/sem/simulation) ──────
def embed_image_clip(image_bytes: bytes, text: str = "") -> list[float]:
    """
    CLIP embedding of image + optional text together.
    When text is provided, combines visual and semantic signals
    in the same vector space.
    """
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # CLIP text encoder max is 77 tokens — truncate reasoning
    if text:
        tokens = clip_processor.tokenizer(
            text,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        inputs = clip_processor(
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        inputs["input_ids"] = tokens["input_ids"]
        inputs["attention_mask"] = tokens["attention_mask"]
    else:
        inputs = clip_processor(
            images=pil_image,
            return_tensors="pt"
        )

    with torch.no_grad():
        if text:
            image_out = clip_model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            text_out = clip_model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )
            image_tensor = image_out if isinstance(image_out, torch.Tensor) else image_out.pooler_output
            text_tensor = text_out if isinstance(text_out, torch.Tensor) else text_out.pooler_output
            features = (image_tensor + text_tensor) / 2
        else:
            image_out = clip_model.get_image_features(
                pixel_values=inputs["pixel_values"]
            )
            features = image_out if isinstance(image_out, torch.Tensor) else image_out.pooler_output

    embedding = features.squeeze().numpy()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()

# ── Single entry point for image embedding ────────────────

def embed_image(image: dict) -> dict | None:
    image_bytes = image["bytes"]
    context = image.get("context", "")

    # Generate Claude reasoning first
    reasoning = generate_reasoning(image_bytes, context)
    image["reasoning"] = reasoning

    # CLIP embedding of image + reasoning together
    clip_embedding = embed_image_clip(image_bytes, text=reasoning)

    return {
        "clip_embedding": clip_embedding,
        "reasoning": reasoning
    }

# ── helper for text only clip embedding ─────────────────────────────
def embed_text_clip(text: str) -> list[float]:
    """
    Embed text alone using CLIP's text encoder.
    Used for searching the image collection with a text question.
    """
    tokens = clip_processor.tokenizer(
        text,
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )
    with torch.no_grad():
        text_out = clip_model.get_text_features(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"]
        )
        text_tensor = text_out if isinstance(text_out, torch.Tensor) else text_out.pooler_output

    embedding = text_tensor.squeeze().numpy()
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()