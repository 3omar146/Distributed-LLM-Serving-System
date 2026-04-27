import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq



print("Initializing Global RAG System...")

gemini_key = "AIzaSyCnSTpdDQLD1MRH9j2yWF8nCqIeaoosRrs"
GROQ_API_KEY="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"


current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "my_bio.txt")

loader = TextLoader(file_path, encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)

retriever = vectorstore.as_retriever()

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context} 

Answer:"""

prompt = PromptTemplate.from_template(template)
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=gemini_key, temperature=0)
llm = ChatGroq(
    temperature=0, 
    model_name="llama-3.1-8b-instant", 
    groq_api_key=GROQ_API_KEY
)

# llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatil", api_key=GROQ_API_KEY)


print("Global RAG System Ready!")