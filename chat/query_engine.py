from langchain_google_genai import ChatGoogleGenerativeAI  # Replace ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from utils.loan_utils import is_loan_query, parse_loan_params, simulate_loan
import os

API_KEY=os.environ.get("GEMINI_API_KEY")
import os
# Load FAISS Vector Store (unchanged)
INDEX_PATH = r"C:\Users\Mega-Pc\Desktop\SAFBOTBACK\faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Replace OpenAI with Gemini
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=API_KEY)  # 👈 Changed
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

def answer_query(query: str) -> str:
    if is_loan_query(query):
        params = parse_loan_params(query)
        return simulate_loan(params)

    result = qa_chain.run(query)
    if not result or result.strip() == "I don't know":
        return llm.invoke(query).content  # Works similarly to OpenAI
    return result