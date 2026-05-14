import os
import uuid
import tempfile
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain_core.documents import Document


router = APIRouter(prefix="/academic", tags=["Academic Intelligence"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PERSISTENT_DIR = os.getenv("PERSISTENT_DIR", ".")
HELPBOT_CHROMA_DIR = os.path.join(PERSISTENT_DIR, "vectorstore", "helpbot")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    api_key=GROQ_API_KEY,
    temperature=0.2,
)

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

session_store: dict[str, dict] = {}


class MessageOut(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    session_id: str
    document: str
    answer: str
    turn: int
    history: list[MessageOut]

class HistoryResponse(BaseModel):
    session_id: str
    document: str
    created_at: str
    last_active: str
    total_turns: int
    history: list[MessageOut]

class SessionSummary(BaseModel):
    session_id: str
    document: str
    created_at: str
    last_active: str
    total_turns: int

ALLOWED_EXTENSIONS = {"pdf", "ppt", "pptx", "txt", "md"}


def _ext(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1] if "." in filename else ""


def _load_document(file_path: str, filename: str) -> list[Document]:
    ext = _ext(filename)
    if ext == "pdf":
        return PyPDFLoader(file_path).load()
    elif ext in ("ppt", "pptx"):
        return UnstructuredPowerPointLoader(file_path).load()
    elif ext in ("txt", "md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [Document(page_content=text, metadata={"source": filename})]
    else:
        raise HTTPException(400, f"Unsupported file type '.{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")


def _index_document(session_id: str, filename: str, file_bytes: bytes) -> tuple[Chroma, int]:
    ext = _ext(filename)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = _load_document(tmp_path, filename)
    finally:
        os.unlink(tmp_path)

    if not docs:
        raise HTTPException(400, "Could not extract any text from the uploaded file.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    collection_name = f"hb_{session_id.replace('-', '')[:36]}"
    persist_dir = os.path.join(HELPBOT_CHROMA_DIR, session_id)
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    return vectorstore, len(chunks)


def _format_history(history: list) -> list[MessageOut]:
    result = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            result.append(MessageOut(role="user", content=msg.content))
        elif isinstance(msg, AIMessage):
            result.append(MessageOut(role="assistant", content=msg.content))
    return result


def _answer(session: dict, question: str) -> str:
    vectorstore: Chroma = session["vectorstore"]
    history: list = session["history"]

    retrieved = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(question)
    context = "\n\n".join(d.page_content for d in retrieved)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a precise academic assistant. Answer the student's question using ONLY "
         "the context from their uploaded document. If the answer is not present, respond "
         "with exactly: \"This information is not in the uploaded document.\"\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "chat_history": history, "question": question})

    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    session["turn_count"] += 1
    session["last_active"] = datetime.utcnow().isoformat()

    return answer


def _get_session_or_404(session_id: str) -> dict:
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(404, f"Session '{session_id}' not found. Start a new session via POST /academic/chat/start")
    return session

@router.post("/chat/start", response_model=ChatResponse, summary="Upload document & ask first question")
async def start_session(
    file: UploadFile = File(..., description="PDF, PPT, PPTX, TXT or MD file"),
    question: str = Form(..., description="Your first question about the document"),
):
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    ext = _ext(file.filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '.{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    session_id = str(uuid.uuid4())
    file_bytes = await file.read()
    vectorstore, chunks = _index_document(session_id, file.filename, file_bytes)

    now = datetime.utcnow().isoformat()
    session = {
        "session_id": session_id,
        "doc_name": file.filename,
        "created_at": now,
        "last_active": now,
        "turn_count": 0,
        "vectorstore": vectorstore,
        "history": [],
        "chunks": chunks,
    }
    session_store[session_id] = session

    answer = _answer(session, question)

    return ChatResponse(
        session_id=session_id,
        document=file.filename,
        answer=answer,
        turn=session["turn_count"],
        history=_format_history(session["history"]),
    )


@router.post("/chat/{session_id}", response_model=ChatResponse, summary="Continue conversation")
async def continue_session(
    session_id: str,
    question: str = Form(..., description="Your follow-up question"),
):
    if not question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    session = _get_session_or_404(session_id)
    answer = _answer(session, question)

    return ChatResponse(
        session_id=session_id,
        document=session["doc_name"],
        answer=answer,
        turn=session["turn_count"],
        history=_format_history(session["history"]),
    )


@router.get("/chat/{session_id}/history", response_model=HistoryResponse, summary="Get full conversation history")
def get_history(session_id: str):
    session = _get_session_or_404(session_id)
    return HistoryResponse(
        session_id=session_id,
        document=session["doc_name"],
        created_at=session["created_at"],
        last_active=session["last_active"],
        total_turns=session["turn_count"],
        history=_format_history(session["history"]),
    )


@router.get("/sessions", response_model=list[SessionSummary], summary="List all active sessions")
def list_sessions():
    return [
        SessionSummary(
            session_id=s["session_id"],
            document=s["doc_name"],
            created_at=s["created_at"],
            last_active=s["last_active"],
            total_turns=s["turn_count"],
        )
        for s in session_store.values()
    ]