# load_csv.py
import csv
from langchain.schema import Document
from typing import List

def load_csv_to_documents(file_path: str) -> List[Document]:
    """Load banking FAQ data from CSV into LangChain Documents."""
    docs = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Combine relevant fields into document content
            content = (
                f"Instruction: {row['instruction']}\n"
                f"Intent: {row['intent']}\n"
                f"Category: {row['category']}\n"
                f"Response: {row['response']}"
            )
            metadata = {
                "tags": row['tags'],
                "category": row['category'],
                "intent": row['intent']
            }
            docs.append(Document(page_content=content, metadata=metadata))
    return docs