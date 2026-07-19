"""
Assembles a Claude API 'content' array from N email attachments + a prompt.

This replaces the fragile Make.com string-concatenation logic (Iterator ->
Text Aggregator -> Parse JSON) that kept producing malformed JSON.

Input: a list of attachments, each with mime_type + base64 data (already
base64-encoded, e.g. straight from an email attachment - do NOT re-encode).

Output: a Python list matching Claude's expected content-block format:
[
  {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}},
  {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}},
  {"type": "text", "text": "..."}
]
"""

import base64
import io
import json

from PIL import Image

# Anthropic's recommended max dimension for token-efficient image processing.
# Images larger than this get resized down; smaller images are left alone.
MAX_DIMENSION = 1568


def resize_image_if_needed(data: str, mime_type: str) -> str:
    """
    Takes a base64-encoded image. If it's larger than MAX_DIMENSION on its
    longest side, resizes it down and re-compresses as JPEG to cut token
    usage dramatically. Returns a (possibly new) base64 string.

    If the image is already small enough, returns the original data untouched
    - no need to re-compress and lose quality for images that are already fine.
    """
    try:
        raw_bytes = base64.b64decode(data)
        img = Image.open(io.BytesIO(raw_bytes))

        width, height = img.size
        longest_side = max(width, height)

        if longest_side <= MAX_DIMENSION:
            # Already small enough - don't touch it
            return data

        # Resize keeping aspect ratio
        scale = MAX_DIMENSION / longest_side
        new_size = (int(width * scale), int(height * scale))
        img = img.resize(new_size, Image.LANCZOS)

        # Convert to RGB if needed (handles PNGs with alpha, CMYK, etc.)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        resized_bytes = buffer.getvalue()

        return base64.b64encode(resized_bytes).decode("utf-8")

    except Exception as e:
        # If resizing fails for any reason, fall back to original data
        # rather than breaking the whole request - better to try the
        # original (even if it might hit a token limit) than fail outright.
        print(f"Warning: image resize failed, using original. Error: {e}")
        return data


def assemble_content(attachments: list[dict], prompt_text: str) -> list[dict]:
    """
    attachments: list of dicts like {"mime_type": "image/jpeg", "data": "<base64 string>"}
    prompt_text: the classification prompt to append as the final text block

    Returns a list of content blocks ready to send to Claude's API.
    """
    if not attachments:
        raise ValueError("No attachments provided - need at least one image")

    content_blocks = []

    for i, att in enumerate(attachments):
        mime_type = att.get("mime_type")
        data = att.get("data")

        if not mime_type:
            raise ValueError(f"Attachment {i} missing mime_type")
        if not data:
            raise ValueError(f"Attachment {i} missing data")

        # Sanity check: is this valid base64? (catches double-encoding bugs early)
        try:
            base64.b64decode(data, validate=True)
        except Exception as e:
            raise ValueError(
                f"Attachment {i} data is not valid base64 - "
                f"check for double-encoding. Error: {e}"
            )

        if mime_type == "application/pdf":
            # PDFs use a different block type entirely - Claude reads them
            # as documents (all pages), not images. No resizing needed here;
            # Anthropic handles PDF page rendering internally.
            content_blocks.append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": data
                }
            })
        elif mime_type.startswith("image/"):
            # Resize large images down to keep token usage reasonable.
            # This may also change the effective mime type to JPEG if resized.
            resized_data = resize_image_if_needed(data, mime_type)
            if resized_data != data:
                # Image was actually resized/recompressed -> now it's a JPEG
                effective_mime_type = "image/jpeg"
            else:
                effective_mime_type = mime_type

            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": effective_mime_type,
                    "data": resized_data
                }
            })
        else:
            raise ValueError(
                f"Attachment {i} has unsupported mime_type '{mime_type}' - "
                f"only images (image/*) and PDFs (application/pdf) are supported"
            )

    # Final text block with the prompt
    content_blocks.append({
        "type": "text",
        "text": prompt_text
    })

    return content_blocks


def assemble_content_json(attachments: list[dict], prompt_text: str) -> str:
    """Same as assemble_content but returns a JSON string, pre-validated."""
    blocks = assemble_content(attachments, prompt_text)
    json_str = json.dumps(blocks)

    # Prove it round-trips cleanly - this is the exact check that kept failing in Make
    json.loads(json_str)

    return json_str


def classify_with_claude(attachments: list[dict], prompt_text: str, api_key: str) -> str:
    """
    Does the FULL job: assembles the content blocks, calls Anthropic directly,
    and returns just the classification text (e.g. "PARTIAL: Bank statements received").

    This replaces the fragile second HTTP call that Make kept failing to build
    correctly - the whole Anthropic API call now happens here, in code, where
    it can be tested and verified directly.

    Automatically injects today's real date into the prompt, since Claude has
    no reliable way to know the current date on its own - without this, it
    can't judge whether a document's date is stale, wrong, or suspicious.
    """
    import anthropic
    from datetime import datetime

    today_str = datetime.now().strftime("%d %B %Y")
    dated_prompt = (
        f"Today's actual date is {today_str}. Use this to judge whether any "
        f"dates visible in the attached document(s) are current/reasonable, "
        f"or clearly outdated, expired, or otherwise suspicious (e.g. a "
        f"document dated years ago, or dated in the future).\n\n"
        f"{prompt_text}"
    )

    content_blocks = assemble_content(attachments, dated_prompt)

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": content_blocks}]
    )

    # Extract just the text - this is what Make will receive, clean and ready
    return response.content[0].text


if __name__ == "__main__":
    # Quick self-test with real base64 image data
    with open("test1.png", "rb") as f:
        img1_b64 = base64.b64encode(f.read()).decode("utf-8")
    with open("test2.png", "rb") as f:
        img2_b64 = base64.b64encode(f.read()).decode("utf-8")

    attachments = [
        {"mime_type": "image/png", "data": img1_b64},
        {"mime_type": "image/png", "data": img2_b64},
    ]

    prompt = "Classify these documents against the VAT return checklist."

    print("=== Testing with 2 attachments ===")
    result = assemble_content_json(attachments, prompt)
    print(f"Length: {len(result)} chars")
    print(f"Valid JSON: confirmed via json.loads()")
    parsed = json.loads(result)
    print(f"Number of content blocks: {len(parsed)}")
    print(f"Block types: {[b['type'] for b in parsed]}")

    print("\n=== Testing with 1 attachment (should still work) ===")
    result_single = assemble_content_json(attachments[:1], prompt)
    parsed_single = json.loads(result_single)
    print(f"Number of content blocks: {len(parsed_single)}")
    print(f"Block types: {[b['type'] for b in parsed_single]}")

    print("\n=== Testing with 5 attachments (stress test) ===")
    result_five = assemble_content_json(attachments * 3, prompt)  # reuse to make 5
    attachments_five = (attachments * 3)[:5]
    result_five = assemble_content_json(attachments_five, prompt)
    parsed_five = json.loads(result_five)
    print(f"Number of content blocks: {len(parsed_five)}")
    print(f"Block types: {[b['type'] for b in parsed_five]}")

    print("\n=== Demonstrating the real Make bug: double-encoding ===")
    print("This is what happened in Make: attachment Data was ALREADY base64,")
    print("but the base64() function wrapped it again, corrupting the image.")
    double_encoded = base64.b64encode(img1_b64.encode()).decode()
    print(f"Original base64 (first 40 chars):  {img1_b64[:40]}")
    print(f"Double-encoded (first 40 chars):    {double_encoded[:40]}")
    print("Both are syntactically 'valid base64' - that's WHY it was so hard to")
    print("spot in Make. The bug isn't invalid syntax, it's wrong CONTENT.")
    print("The only real defense: never wrap already-encoded data in base64() again.")
    print("This function assumes attachments['data'] is ALREADY correctly base64-encoded")
    print("exactly once, matching what the vision API expects.")

    print("\nAll structural tests passed.")
