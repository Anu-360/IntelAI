import os
import pdfplumber
from docx import Document
from PIL import Image
import pytesseract
import pyzmail
import email
from transformers import pipeline
import torch
import cv2
import numpy as np
import warnings
 
# Suppress PDF warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")
 
# Load Hugging Face zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
 
# Default labels
DEFAULT_LABELS = ["invoice", "employment resume", "curriculum vitae", "medical report", "legal contract", "official letter", "email message"]
 
# ===== TEXT EXTRACTION FUNCTIONS =====
 
def extract_text_from_pdf(filepath):
    with pdfplumber.open(filepath) as pdf:
        return "\n".join(page.extract_text() or '' for page in pdf.pages)
 
def extract_text_from_docx(filepath):
    doc = Document(filepath)
    return "\n".join(p.text for p in doc.paragraphs)
 
def preprocess_image_cv(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    temp_path = "temp_preprocessed.png"
    cv2.imwrite(temp_path, thresh)
    return temp_path
 
def extract_text_from_image(image_path):
    preprocessed = preprocess_image_cv(image_path)
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(Image.open(preprocessed), config=custom_config)
 
def extract_text_from_eml(eml_path):
    with open(eml_path, 'rb') as f:
        raw = f.read()
    msg = pyzmail.PyzMessage.factory(raw)
    if msg.text_part:
        return msg.text_part.get_payload().decode(msg.text_part.charset)
    elif msg.html_part:
        return msg.html_part.get_payload().decode(msg.html_part.charset)
    return ""
 
# ===== CLASSIFICATION FUNCTION =====
 
def classify_text_local(text, candidate_labels=None):
    if candidate_labels is None:
        candidate_labels = DEFAULT_LABELS
    result = classifier(text, candidate_labels)
    return result['labels'][0], result
 
# ===== MAIN DOCUMENT CLASSIFIER =====
 
def classify_document(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = extract_text_from_pdf(path)
    elif ext == ".docx":
        text = extract_text_from_docx(path)
    elif ext in [".jpg", ".jpeg", ".png", ".tiff"]:
        text = extract_text_from_image(path)
    elif ext == ".eml":
        text = extract_text_from_eml(path)
    else:
        print(f"❌ Unsupported format: {path}")
        return
 
    if not text.strip():
        print(f"⚠️ No text found in {path}")
        return
 
    print(f"\n📄 Classifying: {path}")
    label, result = classify_text_local(text[:1000])  # Limit to 1000 characters
    print("🔎 Predicted Category:", label)
    print("Top 3 Predictions:", result['labels'][:3])
 
# ===== ENTRY POINT =====
 
if __name__ == "__main__":
    import argparse
 
    parser = argparse.ArgumentParser(description="Classify documents using local transformers")
    parser.add_argument("files", nargs="+", help="Paths to documents (pdf, docx, image, eml)")
 
    args = parser.parse_args()
    for filepath in args.files:
        classify_document(filepath)
 