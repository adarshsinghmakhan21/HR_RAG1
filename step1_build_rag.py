"""
STEP 1 — Build RAG Knowledge Base
Indexes:
- Normal policy content
- Holiday list data
"""

import os
import json
import pickle
import numpy as np
import faiss

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "hr_policies.json"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

print("[INFO] Loading HR policies JSON...")


# ---------------------------------------------------
# LOAD JSON
# ---------------------------------------------------
with open(DATA_FILE, "r", encoding="utf-8") as f:

    docs = json.load(f)

print(f"[INFO] Policies loaded: {len(docs)}")


# ---------------------------------------------------
# CREATE CHUNKS
# ---------------------------------------------------
chunks = []

for doc in docs:

    title = doc.get("title", "Unknown")

    # ---------------------------------------------------
    # NORMAL CONTENT
    # ---------------------------------------------------
    content = doc.get("content", "")

    if content:

        parts = content.split(". ")

        for part in parts:

            text = part.strip()

            if len(text) > 10:

                chunks.append({
                    "text": text,
                    "source": title
                })

    # ---------------------------------------------------
    # HOLIDAY DATA
    # ---------------------------------------------------
    if "holidays" in doc:

        for holiday in doc["holidays"]:

            holiday_name = holiday.get("name", "")
            holiday_date = holiday.get("date", "")
            holiday_day = holiday.get("day", "")

            holiday_text = (
                f"{holiday_name} holiday is on "
                f"{holiday_date} ({holiday_day})"
            )

            chunks.append({
                "text": holiday_text,
                "source": title
            })


print(f"[INFO] Total chunks created: {len(chunks)}")


# ---------------------------------------------------
# CREATE EMBEDDINGS
# ---------------------------------------------------
texts = [c["text"] for c in chunks]

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

X = normalize(
    X.toarray().astype(np.float32)
)


# ---------------------------------------------------
# BUILD FAISS INDEX
# ---------------------------------------------------
dimension = X.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(X)

print(f"[INFO] FAISS index built with {index.ntotal} vectors")


# ---------------------------------------------------
# SAVE FILES
# ---------------------------------------------------
faiss.write_index(
    index,
    os.path.join(MODEL_DIR, "hr_faiss.index")
)

with open(
    os.path.join(MODEL_DIR, "chunks.pkl"),
    "wb"
) as f:

    pickle.dump(chunks, f)

with open(
    os.path.join(MODEL_DIR, "vectorizer.pkl"),
    "wb"
) as f:

    pickle.dump(vectorizer, f)

print("\n✅ RAG knowledge base rebuilt successfully!")