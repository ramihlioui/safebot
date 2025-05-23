import os
import pandas as pd
import re
import csv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

FAQ_PATH = r"C:\Users\Mega-Pc\Desktop\SAFBOTBACK\vectorstore\dataset-banking.csv"
CLEANED_FAQ_PATH = FAQ_PATH.replace('.csv', '_cleaned.csv')
INDEX_PATH = r"C:\Users\Mega-Pc\Desktop\SAFBOTBACK\faiss_index"

def clean_csv(input_path, output_path):
    """Preprocess CSV to ensure proper quoting and delimiter handling."""
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            reader = csv.reader(infile, delimiter=';')
            writer = csv.writer(outfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                cleaned_row = [re.sub(r'\s+', ' ', cell.strip()) for cell in row]
                writer.writerow(cleaned_row)
        print(f"Cleaned CSV saved to: {output_path}")
    except Exception as e:
        print(f"Error cleaning CSV: {str(e)}")
        raise

def inspect_csv_line(file_path, target_line, context_lines=2):
    """Print the problematic line and surrounding lines for debugging."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            start_line = max(1, target_line - context_lines)
            end_line = min(len(lines), target_line + context_lines + 1)
            print(f"Inspecting lines {start_line} to {end_line - 1} around problematic line {target_line}:")
            for i in range(start_line - 1, end_line - 1):
                print(f"Line {i + 1}: {lines[i].strip()}")
    except Exception as e:
        print(f"Error reading file for inspection: {str(e)}")

def validate_csv(df):
    """Validate CSV for profanity, placeholders, and card-specific instructions."""
    profanity = ['fucking', 'fuck', 'shit', 'damn']
    issues = []
    for i, row in df.iterrows():
        instr = row['instruction'].lower()
        resp = row['response'].lower() if isinstance(row['response'], str) else ''
        for word in profanity:
            if word in instr or word in resp:
                issues.append(f"Row {i+2}: Profanity detected ('{word}') in instruction or response")
        if '{{' in resp:
            issues.append(f"Row {i+2}: Placeholder detected in response: {resp[:100]}...")
        if "card" in instr and ("balance" in instr or "solde" in instr or "transaction" in instr):
            issues.append(f"Row {i+2}: Card-specific instruction: {instr[:100]}...")
    return issues

def load_faq_documents(file_path, replace_placeholders=True):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found at: {file_path}")

        df = pd.read_csv(file_path, sep=';', quotechar='"', escapechar='\\', encoding='utf-8')

        required_columns = ['instruction', 'response']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV file must contain all required columns: {required_columns}")

        # Validate CSV
        issues = validate_csv(df)
        if issues:
            print("[DEBUG] CSV validation issues:")
            for issue in issues:
                print(f"[DEBUG] {issue}")

        # Remove duplicates and fix typos
        df['instruction'] = df['instruction'].str.lower().str.strip()
        df['instruction'] = df['instruction'].replace({
            'acivate': 'activate',
            'an card': 'a card'
        }, regex=True)
        df = df.drop_duplicates(subset=['instruction'], keep='first')

        if replace_placeholders:
            df['response'] = df['response'].apply(
                lambda x: re.sub(r'\{\{.*?\}\}', 'account', x) if isinstance(x, str) else x)

        # Log all CSV instructions
        print(f"[DEBUG] Loaded {len(df)} CSV instructions before processing:")
        for i, row in df.iterrows():
            print(f"[DEBUG] CSV Instruction {i+1}: {row['instruction'][:100]}... Response: {row['response'][:100]}...")

        documents = df.apply(lambda row: (
            f"Instruction: {row['instruction']}\nResponse: {row['response']}"
        ), axis=1).tolist()

        print(f"[DEBUG] Loaded {len(documents)} documents from CSV after deduplication")
        for i, doc in enumerate(documents[:5]):
            print(f"[DEBUG] Document {i+1}: {doc[:200]}...")

        return documents
    except pd.errors.ParserError as e:
        error_line = int(str(e).split('line ')[-1].split(',')[0]) if 'line' in str(e) else None
        if error_line:
            print(f"ParserError on line {error_line}:")
            inspect_csv_line(file_path, error_line)
        raise ValueError(
            f" ACHK Error parsing CSV file at {file_path}. Ensure it uses semicolon (;) as delimiter and fields are properly quoted.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file at {file_path} is empty")
    except Exception as e:
        raise Exception(f"Unexpected error processing CSV file: {str(e)}")

def main():
    try:
        if not os.path.exists(CLEANED_FAQ_PATH):
            print(f"[DEBUG] Cleaned CSV not found, cleaning {FAQ_PATH}")
            clean_csv(FAQ_PATH, CLEANED_FAQ_PATH)
        else:
            print(f"[DEBUG] Cleaned CSV found at {CLEANED_FAQ_PATH}, skipping cleaning")

        documents = load_faq_documents(CLEANED_FAQ_PATH, replace_placeholders=True)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\nResponse: ", "\nInstruction: ", "\n"],
            keep_separator=True
        )
        chunks = splitter.split_text("\n\n".join(documents))

        print(f"[DEBUG] Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"[DEBUG] Chunk {i+1}: {chunk[:200]}... (Length: {len(chunk)} characters)")

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)

        os.makedirs(INDEX_PATH, exist_ok=True)
        vectorstore.save_local(INDEX_PATH)
        print(f"✅ FAISS index saved to: {INDEX_PATH}")
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()