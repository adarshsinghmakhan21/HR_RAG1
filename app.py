import streamlit as st

# -------- RAG CODE --------
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Your HR Documents (replace with your actual documents later)
documents = [
    "Employees are entitled to 18 days of paid annual leave per year.",
    "Employees are entitled to 10 sick leave days per year.",
    "Employees can work from home 2 days per week with manager approval.",
    "Working hours are 9 AM to 6 PM, Monday to Friday.",
    "Employees receive health insurance, bonuses, and benefits after probation.",
    "Travel expenses for business trips are reimbursed within 7 days.",
    "Employees must follow company code of conduct and maintain professionalism."
]

# Step 1: Chunking
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.create_documents(documents)

# Step 2: Embeddings
embedding = HuggingFaceEmbeddings()

# Step 3: Vector DB
db = Chroma.from_documents(docs, embedding)

# Step 4: Retriever
retriever = db.as_retriever()

# -------- STREAMLIT UI --------
st.set_page_config(page_title="HR Assistant", layout="centered")

st.title("🤖 HR Policy Assistant")
st.write("Ask any question related to HR policies")

query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if query:
        results = retriever.get_relevant_documents(query)
        
        st.subheader("📌 Answer")
        for doc in results:
            st.write("👉", doc.page_content)

st.caption("Powered by RAG (Retrieval-Augmented Generation)")