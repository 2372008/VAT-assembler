"""
Tiny HTTP API that Make.com will call instead of doing fragile string
concatenation itself.

Make sends: a list of attachments (mime_type + base64 data) + the prompt text.
This returns: the exact JSON array Claude's vision API needs.

Endpoint: POST /assemble
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from assemble import assemble_content

app = FastAPI(title="VAT Document Assembler")


class Attachment(BaseModel):
    mime_type: str
    data: str  # base64-encoded, already encoded exactly once


class AssembleRequest(BaseModel):
    attachments: List[Attachment]
    prompt_text: str


@app.post("/assemble")
def assemble(req: AssembleRequest):
    """
    Make.com calls this with attachments + prompt.
    Returns the content array ready to paste straight into the
    Claude module's Content field (Map mode).
    """
    try:
        attachments_as_dicts = [a.model_dump() for a in req.attachments]
        content = assemble_content(attachments_as_dicts, req.prompt_text)
        return {"content": content}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
def health():
    """Simple check so we know the service is alive."""
    return {"status": "ok"}
