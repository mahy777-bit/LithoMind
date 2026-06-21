import base64
import io
from PIL import Image
from anthropic import Anthropic
from app.config import ANTHROPIC_API_KEY, LLM_MODEL
from app.embeddings import embed_image_clip
from app.vector_interface import search_images

client = Anthropic(api_key=ANTHROPIC_API_KEY)

RECOMMENDER_SYSTEM_PROMPT = """You are LithoMind, a semiconductor lithography and OPC/ILT expert assistant.

The user has provided a layout image and wants a recommendation on whether 
Optical Proximity Correction (OPC) or Inverse Lithography Technology (ILT) 
is more appropriate for their pattern.

You will see the user's layout image directly. You may also receive 
similar layout images found in the literature, along with how those papers 
described or handled them. The user may optionally provide OPC rules or 
design constraints in their description.

You should provide TWO clearly separated parts in your answer:

1. LITERATURE-GROUNDED ANALYSIS: If similar images were found in the literature, 
   compare the user's layout to them and cite sources, e.g. "Similar to [Source 1]...". 
   If no similar literature was found, say so explicitly and skip this part.

2. GENERAL EDGE-BY-EDGE ANALYSIS: Using your general lithography domain knowledge 
   (not the literature above), reason about the layout's edges, corners, and pattern 
   density — and whether shape adjustments would likely be easier to handle globally 
   via ILT or edge-by-edge via OPC. If the user provided specific OPC rules or 
   constraints in their description, factor those in directly. 
   
   Clearly label this section as based on general domain knowledge, not the 
   retrieved literature sources, to keep the two types of reasoning distinct.

Be technically precise and concise. Never blend or confuse which conclusions 
come from literature citations versus general domain reasoning.
"""

def recommend(image_bytes: bytes, description: str = None) -> dict:
    """
    Takes a user's layout image (and optional text description).
    Returns an OPC vs ILT recommendation grounded in visually similar
    literature examples.
    """
    # Step 1: embed the layout image (with description if provided)
    if description:
        query_embedding = embed_image_clip(image_bytes, text=description)
    else:
        query_embedding = embed_image_clip(image_bytes)

    # Step 2: search image_chunks for visually similar literature layouts
    similar_images = search_images(query_embedding, k=3)

    if not similar_images:
        return {
            "recommendation": "I don't have any similar layout examples in my knowledge base to base a recommendation on.",
            "sources": []
        }

    # Step 3: format matched literature for the prompt
    literature_text, sources_list = _format_literature_matches(similar_images)

    # Step 4: send user's image + literature context to Claude
    media_type = _detect_media_type(image_bytes)
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    user_content = [
        {"type": "image", "source": {
            "type": "base64",
            "media_type": media_type,
            "data": b64_image
        }},
        {"type": "text", "text":
            f"This is the user's layout.{' Description: ' + description if description else ''}\n\n"
            f"Similar literature examples:\n{literature_text}\n\n"
            f"Should this layout use OPC or ILT? Explain your reasoning."}
    ]

    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=RECOMMENDER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}]
    )

    return {
        "recommendation": message.content[0].text,
        "sources": sources_list
    }

def _format_literature_matches(similar_images: list) -> tuple[str, list]:
    literature_text = ""
    sources_list = []

    for i, result in enumerate(similar_images):
        label = f"Source {i+1}"
        meta = result["metadata"]
        title = meta.get("title", "Unknown")
        page = meta.get("page", "N/A")

        literature_text += f"[{label}: \"{title}\", page {page}]\n{result['content']}\n\n"

        sources_list.append({
            "label": label,
            "title": title,
            "page": page,
            "paper_id": meta.get("paper_id"),
        })

    return literature_text, sources_list

def _detect_media_type(image_bytes: bytes) -> str:
    pil_image = Image.open(io.BytesIO(image_bytes))
    fmt = pil_image.format.lower() if pil_image.format else "png"
    media_type_map = {
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    return media_type_map.get(fmt, "image/png")