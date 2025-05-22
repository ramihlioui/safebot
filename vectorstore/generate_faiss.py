import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from load_csv import load_csv_to_documents

FAQ_PATH = "vectorstore/faq.csv"
INDEX_PATH = "vectorstore/faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

docs = load_csv_to_documents(FAQ_PATH)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.from_documents(split_docs, embeddings)
vectorstore.save_local(INDEX_PATH)
print("✅ FAISS index saved.")
