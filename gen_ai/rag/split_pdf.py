from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
loader = PyPDFLoader("data/javaproject.pdf")
docs = loader.load()

# Create splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split documents
chunks = text_splitter.split_documents(docs)

print("Number of chunks:", len(chunks))
print("\nFirst chunk:\n")
print(chunks[0].page_content)