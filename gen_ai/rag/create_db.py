import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

folder_path = "documents"

documents = []

for file in os.listdir(folder_path):

    path = os.path.join(folder_path, file)

    try:
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)

        elif file.endswith(".txt"):
            loader = TextLoader(path)

        elif file.endswith(".csv"):
            loader = CSVLoader(path)

        elif file.endswith(".docx"):
            loader = Docx2txtLoader(path)

        elif file.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(path)

        elif file.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(path)

        else:
            print("Skipping:", file)
            continue

        print("Loading:", file)
        documents.extend(loader.load())

    except Exception as e:
        print("Error:", file, e)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="db"
)

print(f"\nTotal Documents Loaded: {len(documents)}")
print(f"Total Chunks Created: {len(chunks)}")
print("Database Created Successfully!")