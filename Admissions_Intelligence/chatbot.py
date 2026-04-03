import os
import json
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

BASE = os.getenv("PERSISTENT_DIR", ".")

VECTOR_BASE = os.path.join(BASE, "vectorstore")
DOCS_BASE   = os.path.join(BASE, "raw_docs")

os.makedirs(VECTOR_BASE, exist_ok=True)
os.makedirs(DOCS_BASE, exist_ok=True)

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    temperature=0.2
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
)

chains = {}
session_stores = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_stores:
        session_stores[session_id] = ChatMessageHistory()
    return session_stores[session_id]


def ingest_pdf(clgcode: str, clg_name: str, pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["clgcode"] = clgcode
        doc.metadata["clg_name"] = clg_name

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    raw_docs_path = os.path.join(DOCS_BASE, f"{clgcode}.json")
    serialised = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]
    with open(raw_docs_path, "w", encoding="utf-8") as f:
        json.dump(serialised, f, ensure_ascii=False, indent=2)

    vector_path = os.path.join(VECTOR_BASE, clgcode)
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=vector_path
    )

    chains.pop(clgcode, None)
    return docs


def load_vectorstore(clgcode: str) -> Chroma:
    vector_path = os.path.join(VECTOR_BASE, clgcode)
    return Chroma(
        persist_directory=vector_path,
        embedding_function=embeddings
    )


def load_raw_docs(clgcode: str):
    from langchain_core.documents import Document
    raw_docs_path = os.path.join(DOCS_BASE, f"{clgcode}.json")
    if not os.path.exists(raw_docs_path):
        return []
    with open(raw_docs_path, "r", encoding="utf-8") as f:
        serialised = json.load(f)
    return [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in serialised
    ]


def format_docs(docs) -> str:
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page", "unknown")
        formatted.append(f"(Page {page})\n{doc.page_content}")
    return "\n\n".join(formatted)


def build_retriever(clgcode: str):
    vectordb = load_vectorstore(clgcode)
    vector_retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20}
    )

    raw_docs = load_raw_docs(clgcode)
    bm25_retriever = BM25Retriever.from_documents(raw_docs)
    bm25_retriever.k = 4

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )
    return hybrid_retriever


def build_chain(clgcode: str):
    retriever = build_retriever(clgcode)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an AI admission assistant for a university.
Answer ONLY from the provided admission brochure context.
Always cite the page number(s) your answer is drawn from, e.g. "(Page 4)".
If the answer cannot be found in the context, say exactly:
"I could not find that information in the brochure. Please contact the university directly."

Context:
{context}"""
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    def retrieval_pipeline(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["question"])
        return {
            "context": format_docs(docs),
            "question": inputs["question"],
            "history": inputs.get("history", []),
        }

    chain = RunnablePassthrough() | retrieval_pipeline | prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )
    return chain_with_history


def get_chain(clgcode: str):
    if clgcode not in chains:
        chains[clgcode] = build_chain(clgcode)
    return chains[clgcode]


def ask_question(clgcode: str, question: str, session_id: str = "default"):
    chain = get_chain(clgcode)
    answer = chain.invoke(
        {"question": question},
        config={"configurable": {"session_id": f"{clgcode}:{session_id}"}}
    )
    return {
        "clgcode": clgcode,
        "session_id": session_id,
        "answer": answer,
    }