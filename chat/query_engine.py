from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from utils.loan_utils import is_loan_query, parse_loan_params, simulate_loan
import os

INDEX_PATH = "vectorstore/faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
api_key = os.getenv("OPENAI_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=api_key)
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def answer_query(query: str) -> str:
    if is_loan_query(query):
        params = parse_loan_params(query)
        return simulate_loan(params)

    result = qa_chain.run(query)
    if not result or result.strip() == "I don't know":
        return llm.invoke(query)
    return result
