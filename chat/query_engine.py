import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils.loan_utils import is_loan_query, parse_loan_params, simulate_loan
import os
import pandas as pd
import re
from autocorrect import Speller
from langdetect import detect
from googletrans import Translator

API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBYZyJ_d9S5fVWHkma1R5xSc1Qiya76TAQ")
CLEANED_FAQ_PATH = r"C:\Users\Mega-Pc\Desktop\SAFBOTBACK\vectorstore\dataset-banking_cleaned.csv"

# Load FAISS Vector Store
INDEX_PATH = r"C:\Users\Mega-Pc\Desktop\SAFBOTBACK\faiss_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
try:
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
except Exception as e:
    print(f"[ERROR] Failed to load FAISS index: {str(e)}")
    retriever = None

# Initialize Gemini model and translator
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)
translator = Translator()

# Tunisian Arabic (Darija) dictionary for typo correction
DARIJA_DICT = {
    "chof": "chouf", "sold": "solde", "sould": "solde", "sald": "solde", "saldy": "solde", "slde": "solde",
    "konto": "compte", "kont": "compte", "kount": "compte", "compt": "compte", "accunt": "compte",
    "bilan": "solde", "balnce": "solde", "blance": "solde",
    "verifi": "vérifier", "chek": "chouf", "tcheki": "chouf", "tshouf": "chouf",
    "ccount": "account", "bnk": "bank", "chck": "check"
}

# Cached CSV instructions
_cached_instructions = None
def load_csv_instructions(file_path):
    """Load instruction fields from CSV for direct lookup."""
    global _cached_instructions
    if _cached_instructions is None:
        try:
            df = pd.read_csv(file_path, sep=';', quotechar='"', encoding='utf-8')
            if 'instruction' not in df.columns or 'response' not in df.columns:
                print("[DEBUG] Missing 'instruction' or 'response' column in CSV")
                return []
            _cached_instructions = df[['instruction', 'response']].to_dict('records')
            print(f"[DEBUG] Loaded {len(_cached_instructions)} CSV instructions:")
            for i, item in enumerate(_cached_instructions):
                print(f"[DEBUG] CSV Instruction {i+1}: {item['instruction'][:100]}...")
        except Exception as e:
            print(f"[DEBUG] Error loading CSV instructions: {str(e)}")
            return []
    return _cached_instructions

def correct_typos(query: str):
    """Correct typos for English, French, and Tunisian Arabic without crashing."""
    corrected = query.lower()
    for wrong, right in DARIJA_DICT.items():
        corrected = corrected.replace(wrong, right)

    try:
        lang = detect(corrected)
        if lang not in ['en', 'fr']:
            lang = 'ar' if any(word in corrected for word in DARIJA_DICT.values()) else 'en'
    except:
        lang = 'en'
    print(f"[DEBUG] Detected language: {lang}")

    if lang == 'en':
        try:
            spell = Speller(lang='en')
            corrected = spell(corrected)
        except Exception as e:
            print(f"[DEBUG] Autocorrect error for English: {str(e)}")
    elif lang == 'fr':
        try:
            spell = Speller(lang='fr')
            corrected = spell(corrected)
        except Exception as e:
            print(f"[DEBUG] Autocorrect error for French: {str(e)}")
    elif lang == 'ar':
        pass

    print(f"[DEBUG] Original query: {query}, Corrected query: {corrected}, Language: {lang}")
    return corrected, lang

def clean_response(response: str) -> str:
    """Clean response by removing placeholders and profanity."""
    response = re.sub(r'\{\{.*?\}\}', 'account', response)
    profanity = ['fucking', 'fuck', 'shit', 'damn']
    for word in profanity:
        response = re.sub(rf'\b{word}\b', '', response, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', response).strip()

def translate_response(response: str, target_lang: str) -> str:
    """Translate response to the target language using synchronous execution."""
    try:
        if target_lang == 'en':
            return response
        elif target_lang == 'fr':
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            translation = loop.run_until_complete(translator.translate(response, dest='fr'))
            loop.close()
            return translation.text
        elif target_lang == 'ar':
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            translation = loop.run_until_complete(translator.translate(response, dest='ar'))
            loop.close()
            translated = translation.text
            darija_translations = {
                "check your current account balance": "chouf solde compte taba3ek",
                "online banking": "banka 3al internet",
                "mobile app": "application mobile",
                "customer support": "dhekma lal clients",
                "account balance": "solde compte",
                "log in": "d5ol",
                "navigate": "cherri",
                "to check your current account balance": "bash tchouf solde compte taba3ek",
                "you have a few options": "3andek barcha khiyarat",
                "visit an atm": "zour ATM",
                "view your recent transactions": "chouf transactions taba3ek l-5irine",
                "to view your recent transactions": "bash tchouf transactions taba3ek l-5irine"
            }
            for eng, dar in darija_translations.items():
                translated = translated.replace(translator.translate(eng, dest='ar').text, dar)
            return translated
        return response
    except Exception as e:
        print(f"[DEBUG] Translation error: {str(e)}")
        return response

def answer_query(query: str) -> str:
    instructions = load_csv_instructions(CLEANED_FAQ_PATH)
    corrected_query, detected_lang = correct_typos(query.strip())

    if is_loan_query(corrected_query):
        params = parse_loan_params(corrected_query)
        return translate_response(simulate_loan(params), detected_lang)

    # CSV lookup for balance and transaction queries
    for item in instructions:
        instr_lower = item['instruction'].lower()
        query_lower = corrected_query.lower()
        # Balance queries
        if ("check" in query_lower and ("balance" in query_lower or "solde" in query_lower) and
            "check" in instr_lower and ("balance" in instr_lower or "solde" in instr_lower) and
            "card" not in instr_lower):
            print(f"[DEBUG] Found matching CSV instruction: {item['instruction']}")
            return translate_response(clean_response(item['response']), detected_lang)
        # Transaction queries
        if (("transaction" in query_lower or "historique" in query_lower or "voir" in query_lower) and
            ("transaction" in instr_lower or "voir" in instr_lower) and
            "card" not in instr_lower):
            print(f"[DEBUG] Found matching CSV instruction: {item['instruction']}")
            return translate_response(clean_response(item['response']), detected_lang)

    # FAISS retrieval for other queries
    if retriever:
        try:
            docs_with_scores = vectorstore.similarity_search_with_score(corrected_query, k=5)
            print(f"[DEBUG] Retrieved {len(docs_with_scores)} documents for query: {corrected_query}")
            for i, (doc, score) in enumerate(docs_with_scores):
                print(f"[DEBUG] Document {i+1} (score: {score:.4f}): {doc.page_content[:200]}...")

            if docs_with_scores:
                top_doc, score = docs_with_scores[0]
                # Relaxed similarity threshold
                if score > 0.5:
                    print(f"[DEBUG] Top document score {score:.4f} too low, using fallback")
                    return translate_response(
                        "Sorry, I couldn't find a relevant answer. Please try rephrasing or contact customer support.",
                        detected_lang
                    )

                response = top_doc.page_content
                # Filter out card-specific or irrelevant responses
                if "card" in response.lower() and ("balance" in corrected_query.lower() or "solde" in corrected_query.lower() or "transaction" in corrected_query.lower()):
                    print(f"[DEBUG] Skipping card-specific response: {response[:200]}...")
                    return translate_response(
                        "Sorry, I couldn't find a relevant answer for your query. Please try rephrasing.",
                        detected_lang
                    )

                if "Response: " in response:
                    response = response.split("Response: ", 1)[-1].strip()
                elif "Instruction: " in response:
                    response = response.split("Instruction: ", 1)[-1].split("\n", 1)[-1].strip()
                if not response:
                    response = "No valid response found for this query."
                print(f"[DEBUG] Selected response (score: {score:.4f}): {response[:200]}...")

                return translate_response(clean_response(response), detected_lang)
        except Exception as e:
            print(f"[DEBUG] FAISS retrieval error: {str(e)}")

    print("[DEBUG] No relevant documents found, using fallback response")
    fallback_response = "Sorry, I couldn't find information for your query. Please try rephrasing or contact customer support."
    return translate_response(fallback_response, detected_lang)