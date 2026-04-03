import json
from PyPDF2 import PdfReader
import fitz
import pytesseract
from PIL import Image
import numpy as np
import cv2
import io
from io import BytesIO
import re

def preprocess_image(pil_img):
    img = np.array(pil_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    return Image.fromarray(gray)


def extract_with_pypdf2(path):
    if isinstance(path, bytes):
        reader = PdfReader(BytesIO(path))
    else:
        reader = PdfReader(path)

    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text.strip()


def extract_with_ocr(path):
    if isinstance(path, bytes):
        doc = fitz.open(stream=path, filetype="pdf")
    else:
        doc = fitz.open(path)

    text = ""

    for i, page in enumerate(doc):
        if i >= 2:
            break

        pix = page.get_pixmap(dpi=400)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img = preprocess_image(img)

        ocr_text = pytesseract.image_to_string(
            img,
            lang="eng+hin",
            config="--oem 3 --psm 6"
        )

        text += ocr_text + "\n"

    return text.strip()


def pdf2text_hybrid(path):
    try:
        text = extract_with_pypdf2(path)
        if len(text) > 100:
            return text
        return extract_with_ocr(path)

    except Exception as e:
        print("HYBRID ERROR:", e)
        return ""


def normalize_text(text):
    return text.lower().strip() if text else ""


def normalize_aadhar(text):
    if not text:
        return ""
    text = re.sub(r"[\s\-]", "", text)
    text = re.sub(r"[^0-9]", "", text)
    return text

REQUIRED_FIELDS = {
    "10th_marksheet": [
        "name",
        "father_name",
        "mother_name",
        "dob",
        "roll_number_10th",
        "board"
    ],

    "12th_marksheet": [
        "name",
        "father_name",
        "mother_name",
        "roll_number_12th",
        "board"
    ],

    "aadhar_card": [
        "name",
        "aadhar_number",
        "vid_number"
    ],

    "entrance_exam": [
        "name",
        "application_number",
        "final_percentile_score"
    ]
}

def match_required_fields(extracted_text, input_fields, required_fields):
    extracted_text_lower = normalize_text(extracted_text)
    matched = {}

    if "aadhar_number" in required_fields or "vid_number" in required_fields:
        aadhar = input_fields.get("aadhar_number")
        vid = input_fields.get("vid_number")

        normalized_text = normalize_aadhar(extracted_text)

        if aadhar:
            normalized_aadhar = normalize_aadhar(aadhar)
            matched["aadhar_number"] = normalized_aadhar in normalized_text

        elif vid:
            normalized_vid = normalize_aadhar(vid)
            matched["vid_number"] = normalized_vid in normalized_text

        if input_fields.get("name"):
            matched["name"] = normalize_text(input_fields["name"]) in extracted_text_lower

        return matched

    for field in required_fields:
        value = input_fields.get(field)
        if not value:
            continue
        matched[field] = normalize_text(value) in extracted_text_lower

    return matched


def verify_documents(uploaded_docs, input_fields):
    results = {}

    for doc_type, file_data in uploaded_docs.items():
        if not file_data:
            continue

        extracted_text = pdf2text_hybrid(file_data)
        required_fields = REQUIRED_FIELDS.get(doc_type, [])

        matched_data = match_required_fields(
            extracted_text,
            input_fields,
            required_fields
        )

        matched_count = sum(matched_data.values())
        total_fields = len(matched_data)

        percentage = round(
            (matched_count / total_fields) * 100, 2
        ) if total_fields else 0.0

        status = "VERIFIED" if percentage >= 80 else "REJECTED"

        results[doc_type] = {
            "parsed_data": extracted_text,
            "matched_data": matched_data,
            "percentage_matched": percentage,
            "verified_status": status
        }

    return results