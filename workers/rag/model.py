import os
import threading
import time
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from groq import RateLimitError

print("Initializing Global RAG System...")

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

keys_string = os.getenv("GROQ_API_KEYS", "").strip()
api_keys = [k.strip() for k in keys_string.split(",") if k.strip()]

if not api_keys:
    print("WARNING: No GROQ_API_KEYS found!")

key_index = 0
key_lock = threading.Lock()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "my_bio.txt")
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

embedding_model = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=HF_TOKEN,
    model="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""

prompt = PromptTemplate.from_template(template)

MAX_LLM_RETRIES = 3
LLM_RETRY_DELAY = 1  # seconds between retries


def get_next_key():
    global key_index
    with key_lock:
        current_key = api_keys[key_index]
        key_index = (key_index + 1) % len(api_keys)
    return current_key


def invoke_llm(formatted_prompt: str) -> str:
    if not api_keys:
        raise ValueError("No API keys available in environment variables.")

    last_error = None

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        current_key = get_next_key()
        print(f"[LLM] Attempt {attempt}/{MAX_LLM_RETRIES} "
              f"using key ...{current_key[-4:]}")

        try:
            llm = ChatGroq(
                temperature=0,
                model_name="llama-3.1-8b-instant",
                groq_api_key=current_key
            )
            return llm.invoke(formatted_prompt).content

        except RateLimitError as e:
            last_error = e
            print(f"[LLM] Rate limit hit on key ...{current_key[-4:]} "
                  f"(attempt {attempt}) — waiting {LLM_RETRY_DELAY}s "
                  f"then trying next key...")
            time.sleep(LLM_RETRY_DELAY)

        except Exception as e:
            print(f"[LLM] Non-rate-limit error on attempt {attempt}: {e}")
            raise e

    raise Exception(f"All {MAX_LLM_RETRIES} LLM attempts failed "
                    f"due to rate limits. Last error: {last_error}")


print("Global RAG System Ready!")