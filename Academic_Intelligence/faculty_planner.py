"""
Academic Intelligence – Faculty Planner
========================================
Two endpoints:

1. POST /academic/faculty/course-planner
   - subject_name, subject_code, num_lectures (required)
   - course_contents: comma-separated topics (optional)
   - file: PDF/PPT/TXT with syllabus (optional)
   At least one of course_contents or file must be provided.

2. POST /academic/faculty/question-paper
   - max_marks, num_objective, num_subjective, difficulty (required)
   - file: PDF/PPT/TXT with notes/content (optional)
   - syllabus: plain-text content (optional)
   At least one of file or syllabus must be provided.
   Returns question paper with answers included.
"""

import os
import re
import json
import tempfile

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


router = APIRouter(prefix="/academic/faculty", tags=["Academic Intelligence"])


llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    api_key=os.getenv("GROQ_API_KEY", ""),
    temperature=0.3,
)


ALLOWED_EXT = {"pdf", "ppt", "pptx", "txt", "md"}


def _ext(filename: str) -> str:
    return filename.lower().rsplit(".", 1)[-1] if "." in filename else ""


def _load_file_text(file_path: str, filename: str) -> str:
    ext = _ext(filename)
    if ext == "pdf":
        docs = PyPDFLoader(file_path).load()
    elif ext in ("ppt", "pptx"):
        docs = UnstructuredPowerPointLoader(file_path).load()
    elif ext in ("txt", "md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise HTTPException(400, f"Unsupported file type '.{ext}'. Allowed: {', '.join(ALLOWED_EXT)}")
    return "\n\n".join(d.page_content for d in docs)


async def _extract_text_from_upload(file: UploadFile) -> str:
    ext = _ext(file.filename)
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"Unsupported file type '.{ext}'. Allowed: {', '.join(ALLOWED_EXT)}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        return _load_file_text(tmp_path, file.filename)
    finally:
        os.unlink(tmp_path)


def _parse_llm_json(raw: str) -> any:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        raw = "\n".join(inner)
    return json.loads(raw.strip())

@router.post("/course-planner", summary="Generate lecture-wise course planner")
async def generate_course_planner(
    subject_name: str = Form(...),
    subject_code: str = Form(...),
    num_lectures: int = Form(...),
    course_contents: Optional[str] = Form(
        default=None,
        description="Comma-separated chapter/topic names (optional if file is provided)"
    ),
    file: Optional[UploadFile] = File(
        default=None,
        description="Syllabus / course outline PDF, PPT or TXT (optional if course_contents is provided)"
    ),
):
    if not re.fullmatch(r"[a-zA-Z0-9]+", subject_code):
        raise HTTPException(400, "subject_code must be alphanumeric only.")
    if not (1 <= num_lectures <= 200):
        raise HTTPException(400, "num_lectures must be between 1 and 200.")

    topics_from_text: list[str] = []
    file_content: str = ""

    if course_contents:
        topics_from_text = [t.strip() for t in course_contents.split(",") if t.strip()]

    if file is not None:
        file_content = await _extract_text_from_upload(file)
        file_content = file_content[:4000]

    if not topics_from_text and not file_content:
        raise HTTPException(400, "Provide at least one of: course_contents or file.")

    topics_line = ", ".join(topics_from_text) if topics_from_text else "Extract from the syllabus content below."
    file_section = f"\nSyllabus file content:\n{file_content}" if file_content else ""

    system_msg = (
        "You are an academic course planner. Generate a lecture-wise course schedule in JSON.\n\n"
        "Rules:\n"
        "- Output ONLY valid JSON. No markdown fences, no explanation.\n"
        "- Distribute topics evenly across the given number of lectures.\n"
        "- Expand broad topics slightly: e.g. 'DSA' → 'DSA - Trees', 'DSA - Graphs'.\n"
        "- Remarks must be very short: 'Introduction', 'Problem solving', 'Revision', 'Guest lecture', etc.\n"
        "- Output schema:\n"
        "  {{\n"
        "    \"subject_name\": string,\n"
        "    \"subject_code\": string,\n"
        "    \"total_lectures\": number,\n"
        "    \"schedule\": [\n"
        "      {{ \"lecture_number\": number, \"topic\": string, \"remarks\": string }}\n"
        "    ]\n"
        "  }}"
    )

    human_msg = (
        "Subject: {subject_name}\n"
        "Code: {subject_code}\n"
        "Topics: {topics}\n"
        "Total Lectures: {num_lectures}"
        "{file_section}\n\n"
        "Generate the course planner JSON."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", human_msg),
    ])

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({
        "subject_name": subject_name,
        "subject_code": subject_code,
        "topics": topics_line,
        "num_lectures": num_lectures,
        "file_section": file_section,
    })

    try:
        return _parse_llm_json(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"LLM returned invalid JSON: {e}. Snippet: {raw[:300]}")


@router.post("/question-paper", summary="Generate question paper with answers")
async def generate_question_paper(
    max_marks: int = Form(...),
    num_objective: int = Form(..., description="Number of MCQ / objective questions"),
    num_subjective: int = Form(..., description="Number of subjective / descriptive questions"),
    difficulty: str = Form(..., description="easy | medium | hard"),
    instructions: Optional[str] = Form(
        default=None,
        description="Any additional instructions for the question paper"
    ),
    file: Optional[UploadFile] = File(
        default=None,
        description="Notes / PPT / PDF to generate questions from (optional if syllabus is provided)"
    ),
    syllabus: Optional[str] = Form(
        default=None,
        description="Plain-text content / topics to base questions on (optional if file is provided)"
    ),
):
    difficulty = difficulty.strip().lower()
    if difficulty not in {"easy", "medium", "hard"}:
        raise HTTPException(400, "difficulty must be: easy, medium, or hard")
    if num_objective < 0 or num_subjective < 0:
        raise HTTPException(400, "num_objective and num_subjective must be >= 0")
    if num_objective + num_subjective < 1:
        raise HTTPException(400, "Total questions (objective + subjective) must be at least 1.")
    if num_objective + num_subjective > 100:
        raise HTTPException(400, "Total questions (objective + subjective) cannot exceed 100.")
    if max_marks < 1:
        raise HTTPException(400, "max_marks must be at least 1.")

    content_parts: list[str] = []

    if file is not None:
        file_text = await _extract_text_from_upload(file)
        if file_text.strip():
            content_parts.append(file_text[:5000])

    if syllabus and syllabus.strip():
        content_parts.append(syllabus.strip()[:2000])

    if not content_parts:
        raise HTTPException(400, "Provide at least one of: file or syllabus.")

    content = "\n\n---\n\n".join(content_parts)

    difficulty_guidance = {
        "easy":   "Questions should be straightforward recall and basic understanding. Avoid tricky wording.",
        "medium": "Questions should require moderate understanding and application of concepts.",
        "hard":   "Questions should be challenging, requiring deep understanding, analysis, and multi-step reasoning.",
    }[difficulty]

    total_questions = num_objective + num_subjective

    if num_objective > 0 and num_subjective > 0:
        q_desc = (
            f"Generate exactly {num_objective} MCQ questions and {num_subjective} subjective questions. "
            "MCQs must have 4 options (A, B, C, D), a correct_answer (letter only), and a brief explanation. "
            "Subjective questions must include a detailed model_answer."
        )
        schema = (
            "Each question object: question_number (int), type ('mcq' or 'subjective'), question (str), "
            "options (object with keys A B C D — mcq only), correct_answer (str — mcq only), "
            "explanation (str — mcq only), model_answer (str — subjective only), marks (int)"
        )
    elif num_objective > 0:
        q_desc = (
            f"Generate exactly {num_objective} MCQ questions. "
            "Each must have 4 options (A, B, C, D), a correct_answer (letter only), and a brief explanation."
        )
        schema = (
            "Each question object: question_number (int), type = 'mcq', question (str), "
            "options (object with keys A B C D), correct_answer (str), explanation (str), marks (int)"
        )
    else:
        q_desc = (
            f"Generate exactly {num_subjective} subjective/descriptive questions. "
            "Each must include a detailed model_answer."
        )
        schema = (
            "Each question object: question_number (int), type = 'subjective', question (str), "
            "model_answer (str), marks (int)"
        )

    extra = (
        f"\nAdditional faculty instructions (must be strictly followed):\n{instructions.strip()}"
        if instructions and instructions.strip() else ""
    )

    system_msg = (
        "You are an expert academic question paper setter.\n\n"
        "Rules:\n"
        "- Output ONLY valid JSON. No markdown fences, no extra text.\n"
        "- Base ALL questions strictly on the provided content.\n"
        "- Marks across all questions must sum to exactly the given max_marks.\n"
        "- Include complete answers for every question.\n"
        f"- Difficulty level: {difficulty.upper()} — {difficulty_guidance}"
        f"{extra}\n\n"
        "Output schema:\n"
        "  title (str), total_marks (int), total_questions (int), difficulty (str),\n"
        "  questions (array) where:\n"
        f"  {schema}"
    )

    human_msg = (
        "Content:\n{content}\n\n"
        "Instructions: {q_desc}\n"
        "Max Marks: {max_marks}\n"
        "Total Questions: {total_questions}\n\n"
        "Generate the question paper JSON now."
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("human", human_msg),
    ])

    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({
        "content": content,
        "q_desc": q_desc,
        "max_marks": max_marks,
        "total_questions": total_questions,
    })

    try:
        return _parse_llm_json(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(500, f"LLM returned invalid JSON: {e}. Snippet: {raw[:300]}")