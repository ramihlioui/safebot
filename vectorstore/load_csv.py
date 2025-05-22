import csv
from langchain.schema import Document

def load_csv_to_documents(file_path: str):
    docs = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            content = f"Instruction: {row['instruction']}\nIntent: {row['intent']}\nResponse: {row['response']}"
            docs.append(Document(page_content=content))
    return docs
