import os
import json
import re

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from Admissions_Intelligence.stringmatching import verify_documents
from Admissions_Intelligence.chatbot import ingest_pdf, ask_question
from Academic_Intelligence.help_bot import router as helpbot_router
from Academic_Intelligence.faculty_planner import router as faculty_router

app = FastAPI(
    title="Campus One AI",
    description="Admissions Intelligence + Academic Intelligence API",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Admissions Intelligence",
            "description": "Document verification, brochure ingestion, and RAG-powered college chatbot.",
        },
        {
            "name": "Academic Intelligence",
            "description": "Student help bot (document Q&A with session history) and faculty tools (course planner, question paper generator).",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(helpbot_router)
app.include_router(faculty_router)

BASE = os.getenv("PERSISTENT_DIR", ".")
UPLOAD_DIR = os.path.join(BASE, "uploaded_docs")
REGISTRY_PATH = os.path.join(BASE, "registry.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)


def load_registry() -> dict:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}


def save_registry(reg: dict):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(reg, f, indent=2)


registry = load_registry()


@app.get("/", tags=["Admissions Intelligence"])
def home():
    return {
        "status": "ok",
        "message": "Campus One AI API is live",
    }


@app.post("/verify", tags=["Admissions Intelligence"])
async def verify_documents_api(
    documents: List[UploadFile] = File(...),
    doc_types: str = Form(...),
    input_fields: str = Form(...),
):
    try:
        input_fields = json.loads(input_fields)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in input_fields")

    doc_types_list = [d.strip() for d in doc_types.split(",") if d.strip()]
    if len(documents) != len(doc_types_list):
        raise HTTPException(400, "documents and doc_types count mismatch")

    uploaded_docs = {}
    for doc, doc_type in zip(documents, doc_types_list):
        if doc.content_type != "application/pdf":
            raise HTTPException(400, "Only PDF files allowed")
        uploaded_docs[doc_type] = await doc.read()

    return verify_documents(uploaded_docs, input_fields)


@app.post("/upload-brochure", tags=["Admissions Intelligence"])
async def upload_brochure(
    clgcode: str = Form(...),
    clg_name: str = Form(...),
    file: UploadFile = File(...),
):
    if not re.fullmatch(r"[a-zA-Z0-9]+", clgcode):
        raise HTTPException(400, "clgcode must be alphanumeric only")
    if clgcode in registry:
        raise HTTPException(409, f"Brochure for '{clgcode}' already exists. Use /update-brochure to replace.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")

    path = os.path.join(UPLOAD_DIR, f"{clgcode}.pdf")
    with open(path, "wb") as f:
        f.write(await file.read())

    docs = ingest_pdf(clgcode, clg_name, path)
    registry[clgcode] = clg_name
    save_registry(registry)

    return {
        "status": "brochure uploaded and indexed",
        "clgcode": clgcode,
        "clg_name": clg_name,
        "chunks_created": len(docs),
    }


@app.post("/update-brochure", tags=["Admissions Intelligence"])
async def update_brochure(
    clgcode: str = Form(...),
    clg_name: str = Form(...),
    file: UploadFile = File(...),
):
    if not re.fullmatch(r"[a-zA-Z0-9]+", clgcode):
        raise HTTPException(400, "clgcode must be alphanumeric only")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files allowed")

    path = os.path.join(UPLOAD_DIR, f"{clgcode}.pdf")
    with open(path, "wb") as f:
        f.write(await file.read())

    docs = ingest_pdf(clgcode, clg_name, path)
    registry[clgcode] = clg_name
    save_registry(registry)

    return {
        "status": "brochure updated and re-indexed",
        "clgcode": clgcode,
        "clg_name": clg_name,
        "chunks_created": len(docs),
    }


@app.get("/colleges", tags=["Admissions Intelligence"])
def list_colleges():
    return registry


@app.post("/chat", tags=["Admissions Intelligence"])
async def chat(
    clgcode: str = Form(...),
    question: str = Form(...),
    session_id: str = Form(default="default"),
):
    if not re.fullmatch(r"[a-zA-Z0-9]+", clgcode):
        raise HTTPException(400, "clgcode must be alphanumeric only")
    if clgcode not in registry:
        raise HTTPException(404, f"College '{clgcode}' not found. Upload brochure first.")
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty")

    return ask_question(clgcode, question, session_id)