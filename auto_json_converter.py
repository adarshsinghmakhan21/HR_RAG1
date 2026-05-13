import os
import json
import pdfplumber
import pandas as pd
from docx import Document

UPLOAD_FOLDER = "uploads"
JSON_FILE = "data/hr_policies.json"


# ---------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------
def extract_pdf(path):

    text = ""

    try:
        with pdfplumber.open(path) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        print(f"PDF Error ({path}): {e}")

    return text.strip()


# ---------------------------------------------------
# DOCX TEXT EXTRACTION
# ---------------------------------------------------
def extract_docx(path):

    text = ""

    try:
        doc = Document(path)

        text = "\n".join(
            [para.text for para in doc.paragraphs]
        )

    except Exception as e:
        print(f"DOCX Error ({path}): {e}")

    return text.strip()


# ---------------------------------------------------
# XLSX TEXT EXTRACTION
# ---------------------------------------------------
def extract_xlsx(path):

    text = ""

    try:
        excel = pd.ExcelFile(path)

        for sheet in excel.sheet_names:

            df = excel.parse(sheet)

            text += df.to_string(index=False)
            text += "\n"

    except Exception as e:
        print(f"XLSX Error ({path}): {e}")

    return text.strip()


# ---------------------------------------------------
# LOAD EXISTING JSON
# ---------------------------------------------------
def load_json():

    if not os.path.exists(JSON_FILE):
        return []

    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception as e:
        print(f"JSON Load Error: {e}")
        return []


# ---------------------------------------------------
# SAVE JSON
# ---------------------------------------------------
def save_json(data):

    try:
        with open(JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False
            )

    except Exception as e:
        print(f"JSON Save Error: {e}")


# ---------------------------------------------------
# MAIN PROCESS
# ---------------------------------------------------
def process_uploads():

    print("\n[INFO] Starting upload processing...\n")

    # Create upload folder if missing
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    existing_docs = load_json()

    processed_titles = {
        doc["title"].lower()
        for doc in existing_docs
    }

    added_count = 0

    files = os.listdir(UPLOAD_FOLDER)

    if not files:
        print("[INFO] No files found in uploads/")
        return

    for filename in files:

        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        # Skip folders
        if os.path.isdir(filepath):
            continue

        title = os.path.splitext(filename)[0]

        filename_lower = filename.lower()

        # Duplicate check
        if title.lower() in processed_titles:
            print(f"Skipping existing file: {filename}")
            continue

        content = ""

        # ---------------------------------------------------
        # PDF
        # ---------------------------------------------------
        if filename_lower.endswith(".pdf"):

            print(f"[PDF] Processing: {filename}")

            content = extract_pdf(filepath)

        # ---------------------------------------------------
        # DOCX
        # ---------------------------------------------------
        elif filename_lower.endswith(".docx"):

            print(f"[DOCX] Processing: {filename}")

            content = extract_docx(filepath)

        # ---------------------------------------------------
        # XLSX
        # ---------------------------------------------------
        elif filename_lower.endswith(".xlsx"):

            print(f"[XLSX] Processing: {filename}")

            content = extract_xlsx(filepath)

        # ---------------------------------------------------
        # Unsupported
        # ---------------------------------------------------
        else:

            print(f"Unsupported file skipped: {filename}")
            continue

        # Empty file check
        if not content.strip():

            print(f"Empty content skipped: {filename}")
            continue

        # Create document
        new_doc = {
            "id": f"doc{len(existing_docs) + 1}",
            "title": title,
            "content": content
        }

        existing_docs.append(new_doc)

        processed_titles.add(title.lower())

        added_count += 1

        print(f"Added: {filename}")

    # Save updated JSON
    save_json(existing_docs)

    print("\n===================================")
    print("JSON Updated Successfully ✅")
    print(f"New Documents Added: {added_count}")
    print(f"Total Policies: {len(existing_docs)}")
    print("===================================\n")


# ---------------------------------------------------
# RUN
# ---------------------------------------------------
if __name__ == "__main__":

    process_uploads()