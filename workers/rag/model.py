import os
import requests
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

# ---- Document loading + embeddings (unchanged) ----
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "my_bio.txt")
loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

embedding_model = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=HF_TOKEN,
    model="sentence-transformers/all-MiniLM-L6-v2",
)
vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# ---- Prompt template (unchanged) ----
template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""

prompt = PromptTemplate.from_template(template)


def invoke_llm(formatted_prompt: str) -> dict:
    """
    Send the prompt to this worker's assigned GPU server.
    Returns the full response dict (answer + metrics + token counts).
    Raises on HTTP errors or non-OK status so the worker's try/except can mark the request failed.
    """
    if not GPU_SERVER_URL:
        raise RuntimeError("GPU_SERVER_URL not configured")

    res = requests.post(
        f"{GPU_SERVER_URL}/generate",
        json={"prompt": formatted_prompt, "max_new_tokens": MAX_NEW_TOKENS},
        timeout=240,
    )
    res.raise_for_status()
    data = res.json()

    if not data.get("ok"):
        raise RuntimeError(f"GPU server returned not-ok: {data}")

    return data


print("Global RAG System Ready!")