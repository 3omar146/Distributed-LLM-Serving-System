import os
import threading
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

print("Initializing Global RAG System...")

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# 1. Grab keys into a STATIC list (no popping!)
keys_string = os.getenv("GROQ_API_KEYS", "").strip()
api_keys = [k.strip() for k in keys_string.split(",") if k.strip()]

if not api_keys:
    print("WARNING: No GROQ_API_KEYS found!")

# We use a thread lock and a counter for safe rotation
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

# 2. The Bulletproof Rotation Function
def invoke_llm(formatted_prompt: str) -> str:
    global key_index
    
    if not api_keys:
        raise ValueError("No API keys available in environment variables.")
        
    # Safely get the next key using modulo math
    with key_lock:
        current_key = api_keys[key_index]
        key_index = (key_index + 1) % len(api_keys)
        
    print(f"Using API Key ending in: ...{current_key[-4:]}")
    
    llm = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
        groq_api_key=current_key
    )
    
    return llm.invoke(formatted_prompt).content

print("Global RAG System Ready!")