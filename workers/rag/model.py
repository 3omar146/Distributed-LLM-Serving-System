import os
import requests
import concurrent.futures
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate

print("Initializing Global RAG System...")

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
GPU_SERVER_URL = os.getenv("GPU_SERVER_URL", "").strip()
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 150))

if not GPU_SERVER_URL:
    print("WARNING: GPU_SERVER_URL is not set!")
else:
    print(f"[RAG] Worker will send LLM requests to: {GPU_SERVER_URL}")

# ---- Document loading + embeddings ----
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

# ---- Prompt template ----
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:"""
prompt = PromptTemplate.from_template(template)

def invoke_llm(formatted_prompt: str) -> dict:
    """Original Single Request Function"""
    if not GPU_SERVER_URL:
        raise RuntimeError("GPU_SERVER_URL not configured")
    res = requests.post(
        f"{GPU_SERVER_URL}/generate",
        json={"prompt": formatted_prompt, "max_new_tokens": MAX_NEW_TOKENS},
        timeout=240,
    )
    res.raise_for_status()
    data = res.json()
    if not data.get("ok", True) and "answer" not in data:
        raise RuntimeError(f"GPU server returned not-ok: {data}")
    return data

def invoke_llm_batch(formatted_prompts: list) -> list:
    """
    DYNAMIC BATCHING: Sends multiple prompts to the GPU at once.
    Includes a fallback in case the GPU server only accepts strings.
    """
    if not GPU_SERVER_URL:
        raise RuntimeError("GPU_SERVER_URL not configured")
    
    try:
        # Attempt True GPU Batching (Sending a list)
        res = requests.post(
            f"{GPU_SERVER_URL}/generate",
            json={"prompt": formatted_prompts, "max_new_tokens": MAX_NEW_TOKENS},
            timeout=240,
        )
        res.raise_for_status()
        data = res.json()
        
        # If the GPU server natively supports and returns a list of results
        if isinstance(data, list):
            return data
            
    except Exception as e:
        print(f"[RAG] Native GPU batching failed/unsupported. Using concurrent fallback...")

    # FALLBACK: If GPU server doesn't support arrays, blast the requests 
    # concurrently from a background thread pool to avoid locking up FastAPI.
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(formatted_prompts)) as executor:
        futures = [executor.submit(invoke_llm, p) for p in formatted_prompts]
        for future in futures:
            try:
                results.append(future.result())
            except Exception as exc:
                results.append({"answer": f"Error: {exc}", "metrics": {}})
    return results

print("Global RAG System Ready!")