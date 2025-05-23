import csv
from langchain.schema import Document
from typing import List

FAQ_PATH = r"C:\Users\Mega-Pc\Desktop\SAFBOTBACK\vectorstore\faq.csv"
CLEANED_FAQ_PATH = FAQ_PATH.replace('.csv', '_cleaned.csv')


def clean_csv(input_path, output_path):
    """Preprocess CSV to ensure proper quoting and delimiter handling."""
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8',
                                                                     newline='') as outfile:
            reader = csv.reader(infile, delimiter=';')
            writer = csv.writer(outfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                writer.writerow(row)
        print(f"Cleaned CSV saved to: {output_path}")
    except Exception as e:
        print(f"Error cleaning CSV: {str(e)}")
        raise


def load_csv_to_documents(file_path: str) -> List[Document]:
    """Load banking FAQ data from CSV into LangChain Documents."""
    try:
        # Clean the CSV first
        clean_csv(file_path, CLEANED_FAQ_PATH)
        file_path = CLEANED_FAQ_PATH  # Use cleaned CSV

        docs = []
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            # Verify required columns
            required_columns = ['instruction','response']
            if not all(col in reader.fieldnames for col in required_columns):
                raise ValueError(f"CSV file must contain all required columns: {required_columns}")

            for row in reader:
                # Combine relevant fields into document content
                content = (
                    f"Instruction: {row['instruction']}\n"
                    f"Response: {row['response']}"
                )
                docs.append(Document(page_content=content))
        return docs
    except Exception as e:
        print(f"Error loading CSV to documents: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage
    documents = load_csv_to_documents(FAQ_PATH)
    print(f"Loaded {len(documents)} documents")