import os
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi

import rag   # our RAG logic lives in rag.py

app = FastAPI(title="RAG Chatbot")


# ------------------------------------------------------------------
# Request / response shapes (Pydantic checks incoming JSON against these)
# ------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    question: str
    user_name: str = "Friend"      # optional


class Source(BaseModel):
    filename: str
    page: Optional[int] = None
    score: float
    preview: str


class ChatResponse(BaseModel):
    answer: str
    route: str
    search_question: Optional[str] = None
    sources: List[Source]

# Make Swagger UI show file pickers (newer FastAPI describes files in a way Swagger UI can't draw)
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(title=app.title, version=app.version,
                         openapi_version=app.openapi_version, routes=app.routes)

    for body in schema.get("components", {}).get("schemas", {}).values():
        for prop in body.get("properties", {}).values():
            for node in (prop, prop.get("items", {})):
                if node.get("contentMediaType") == "application/octet-stream":
                    node.pop("contentMediaType")
                    node["format"] = "binary"

    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


# ------------------------------------------------------------------
# Endpoint 1: INGEST - upload one or more files
# (plain "def", not "async def": embedding calls are blocking,
#  FastAPI runs plain def endpoints in a worker thread so the server stays responsive)
# ------------------------------------------------------------------
@app.post("/ingest")
def ingest(files: List[UploadFile] = File(...)):
    os.makedirs(rag.FILES_FOLDER, exist_ok=True)
    results = []

    for upload in files:
        filename = os.path.basename(upload.filename)    # basename blocks "../" tricks
        ext = os.path.splitext(filename)[1].lower()

        if ext not in rag.SUPPORTED:
            results.append({"filename": filename, "status": "skipped",
                            "detail": f"unsupported file type {ext}"})
            continue

        path = os.path.join(rag.FILES_FOLDER, filename)
        with open(path, "wb") as f:
            f.write(upload.file.read())

        try:
            chunk_count = rag.ingest_file(path)
            results.append({"filename": filename, "status": "ingested", "chunks": chunk_count})
        except Exception as e:
            results.append({"filename": filename, "status": "failed", "detail": str(e)})

    return {"results": results}


# ------------------------------------------------------------------
# Endpoint 2: CHAT - ask a question
# ------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")
    

    return rag.ask(request.session_id, request.user_name, request.question)