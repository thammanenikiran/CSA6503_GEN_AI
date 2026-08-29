from dotenv import load_dotenv
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load API key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# Load PDF
loader = PyPDFLoader("data/javaproject.pdf")
docs = loader.load()

# Split PDF
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print("Chunks created:", len(chunks))

# Create embedding model
embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# Store in ChromaDB
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="db"
)

print("Embeddings stored successfully!")