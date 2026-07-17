"""
Tiny HTTP API that Make.com calls instead of doing fragile string
concatenation and separate Anthropic API calls itself.

Two endpoints:
- /assemble : builds the content array only (legacy, kept for reference)
- /classify : does the FULL job - assembles attachments, calls Anthropic
              directly, returns just the clean classification text.
              This is the one Make should actually use.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os

from assemble import assemble_content, classify_with_claude

app = FastAPI(title="VAT Document Assembler")


class Attachment(BaseModel):
    mime_type: str
    data: str  # base64-encoded, already encoded exactly once


class AssembleRequest(BaseModel):
    attachments: List[Attachment]
    prompt_text: str


class ClassifyRequest(BaseModel):
    attachments: List[Attachment]
    prompt_text: str


@app.post("/assemble")
def assemble(req: AssembleRequest):
    """
    Legacy endpoint - builds the content array only, doesn't call Anthropic.
    Kept for reference/debugging. Make should use /classify instead.
    """
    try:
        attachments_as_dicts = [a.model_dump() for a in req.attachments]
        content = assemble_content(attachments_as_dicts, req.prompt_text)
        return {"content": content}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/classify")
def classify(req: ClassifyRequest):
    """
    Does the FULL job: assembles attachments (resizing images as needed),
    calls Anthropic directly, returns just the classification text.

    Make only needs to call this ONE endpoint with attachments + prompt_text
    and gets back a clean result string ready to write into Google Sheets.
    No separate Anthropic API call needed on Make's side anymore.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured on server")

    try:
        attachments_as_dicts = [a.model_dump() for a in req.attachments]
        result_text = classify_with_claude(attachments_as_dicts, req.prompt_text, api_key)
        return {"result": result_text}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@app.get("/health")
def health():
    """Simple check so we know the service is alive."""
    return {"status": "ok"}
