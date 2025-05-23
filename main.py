from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from chat.query_engine import answer_query

app = FastAPI(title="AI Banking Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=[ "Content-Type"],
)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: QueryRequest):
    print(req)
    response = answer_query(req.question.strip())
    print(response)
    return {"question": req.question, "response": response}
