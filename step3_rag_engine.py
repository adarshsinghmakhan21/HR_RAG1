"""
STEP 3 — HR POLICY RAG ENGINE
Single Best Answer Version + Holiday Intelligence
"""

import os
import json
import pickle
import numpy as np
import faiss

from datetime import datetime
from sklearn.preprocessing import normalize


# ---------------------------------------------------
# PATHS
# ---------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)


# ---------------------------------------------------
# MAIN RAG CLASS
# ---------------------------------------------------
class HRPolicyRAG:

    def __init__(self):

        print("\n[STEP 3] Initializing HR RAG Engine...\n")

        # ---------------------------------------------------
        # LOAD FAISS INDEX
        # ---------------------------------------------------
        self.index = faiss.read_index(
            os.path.join(
                MODEL_DIR,
                "hr_faiss.index"
            )
        )

        # ---------------------------------------------------
        # LOAD CHUNKS
        # ---------------------------------------------------
        with open(
            os.path.join(
                MODEL_DIR,
                "chunks.pkl"
            ),
            "rb"
        ) as f:

            self.chunks = pickle.load(f)

        # ---------------------------------------------------
        # LOAD VECTORIZER
        # ---------------------------------------------------
        with open(
            os.path.join(
                MODEL_DIR,
                "vectorizer.pkl"
            ),
            "rb"
        ) as f:

            self.vectorizer = pickle.load(f)

        # ---------------------------------------------------
        # LOAD POLICIES
        # ---------------------------------------------------
        with open(
            os.path.join(
                DATA_DIR,
                "hr_policies.json"
            ),
            "r",
            encoding="utf-8"
        ) as f:

            self.docs = json.load(f)

        print(f"[INFO] Loaded chunks: {len(self.chunks)}")
        print(f"[INFO] Loaded policies: {len(self.docs)}")
        print(f"[INFO] FAISS vectors: {self.index.ntotal}")

        print("\n✅ RAG Engine Ready\n")

    # ---------------------------------------------------
    # RETRIEVE BEST MATCH
    # ---------------------------------------------------
    def retrieve(self, query, top_k=1):

        q_vec = self.vectorizer.transform([query])

        q_vec = normalize(
            q_vec.toarray().astype(np.float32)
        )

        scores, ids = self.index.search(
            q_vec,
            k=top_k
        )

        results = []

        for score, idx in zip(scores[0], ids[0]):

            if idx < len(self.chunks):

                results.append({
                    "text": self.chunks[idx]["text"],
                    "source": self.chunks[idx]["source"],
                    "score": float(score)
                })

        return results

    # ---------------------------------------------------
    # INTENT DETECTION
    # ---------------------------------------------------
    def detect_intent(self, query):

        q = query.lower()

        if "leave" in q:
            return "leave_policy"

        elif "salary" in q:
            return "salary_policy"

        elif "attendance" in q:
            return "attendance_policy"

        elif "work from home" in q:
            return "wfh_policy"

        elif "security" in q:
            return "security_policy"

        elif "email" in q:
            return "email_policy"

        elif "password" in q:
            return "password_policy"

        elif "holiday" in q:
            return "holiday_policy"

        else:
            return "general_policy"

    # ---------------------------------------------------
    # GENERATE SINGLE ANSWER
    # ---------------------------------------------------
    def generate_answer(self, retrieved):

        if not retrieved:
            return "No relevant HR policy found."

        best = retrieved[0]

        return best["text"]

    # ---------------------------------------------------
    # MAIN ASK FUNCTION
    # ---------------------------------------------------
    def ask(self, query):

        try:

            q = query.lower()

            # =================================================
            # NEXT HOLIDAY LOGIC
            # =================================================
            if "next holiday" in q or "upcoming holiday" in q:

                today = datetime.today()

                holiday_doc = None

                for doc in self.docs:

                    if doc.get("type") == "holiday_data":
                        holiday_doc = doc
                        break

                if holiday_doc:

                    upcoming = []

                    for holiday in holiday_doc.get("holidays", []):

                        try:

                            holiday_date_str = holiday.get(
                                "date",
                                ""
                            )

                            # Skip invalid dates
                            if len(holiday_date_str) != 10:
                                continue

                            holiday_date = datetime.strptime(
                                holiday_date_str,
                                "%Y-%m-%d"
                            )

                            if holiday_date >= today:

                                upcoming.append({
                                    "name": holiday["name"],
                                    "date": holiday["date"],
                                    "day": holiday.get("day", ""),
                                    "datetime": holiday_date
                                })

                        except Exception as e:

                            print(
                                "Holiday Parsing Error:",
                                e
                            )

                            continue

                    # Sort nearest holiday
                    upcoming.sort(
                        key=lambda x: x["datetime"]
                    )

                    if upcoming:

                        next_holiday = upcoming[0]

                        return {
                            "query": query,
                            "intent": "holiday_policy",
                            "answer":
                                f"Your next holiday is "
                                f"{next_holiday['name']} on "
                                f"{next_holiday['date']} "
                                f"({next_holiday['day']})",
                            "sources": ["Holiday List 2026"],
                            "top_chunks": []
                        }

            # =================================================
            # SPECIFIC HOLIDAY SEARCH
            # =================================================
            if "holiday" in q:

                for doc in self.docs:

                    if doc.get("type") == "holiday_data":

                        holidays = doc.get(
                            "holidays",
                            []
                        )

                        for holiday in holidays:

                            holiday_name = holiday[
                                "name"
                            ].lower()

                            if holiday_name.split()[0] in q:

                                return {
                                    "query": query,
                                    "intent": "holiday_policy",
                                    "answer":
                                        f"{holiday['name']} "
                                        f"holiday is on "
                                        f"{holiday['date']} "
                                        f"({holiday['day']})",
                                    "sources": [doc["title"]],
                                    "top_chunks": []
                                }

            # =================================================
            # NORMAL RAG SEARCH
            # =================================================
            retrieved = self.retrieve(
                query=query,
                top_k=1
            )

            answer = self.generate_answer(
                retrieved
            )

            return {
                "query": query,
                "intent": self.detect_intent(query),
                "answer": answer,
                "sources": list({
                    r["source"]
                    for r in retrieved
                }),
                "top_chunks": retrieved
            }

        except Exception as e:

            return {
                "query": query,
                "intent": "error",
                "answer": f"Error: {str(e)}",
                "sources": [],
                "top_chunks": []
            }


# ---------------------------------------------------
# TEST MODE
# ---------------------------------------------------
if __name__ == "__main__":

    rag = HRPolicyRAG()

    print("=" * 60)
    print(" HR POLICY RAG ASSISTANT ")
    print("=" * 60)

    while True:

        q = input("\nAsk Question: ")

        if q.lower() == "exit":
            break

        result = rag.ask(q)

        print("\nANSWER:\n")
        print(result["answer"])

        print("\nINTENT:")
        print(result["intent"])

        print("\nSOURCES:")
        print(", ".join(result["sources"]))

        print("\n" + "-" * 60)