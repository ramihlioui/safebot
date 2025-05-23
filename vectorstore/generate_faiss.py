import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAQ_PATH = r"C:\Users\Mega-Pc\Desktop\SAFBOTBACK\vectorstore\faq.txt"
INDEX_PATH = "faiss_index"

def load_faq_documents(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    text = load_faq_documents(FAQ_PATH)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    os.makedirs(INDEX_PATH, exist_ok=True)
    vectorstore.save_local(INDEX_PATH)
    print(f"✅ FAISS index saved to: {INDEX_PATH}")

if __name__ == "__main__":
    main()
