from dotenv import load_dotenv
import os

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma import Chroma

# ===========================
# Load API Key
# ===========================
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# ===========================
# Load Embedding Model
# ===========================
embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

# ===========================
# Load Chroma Database
# ===========================
db = Chroma(
    persist_directory="db",
    embedding_function=embedding
)

# ===========================
# Load Gemini Model
# ===========================
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key,
   
)

print("=" * 50)
print("      PDF QUESTION ANSWERING SYSTEM")
print("=" * 50)

while True:

    question = input("\nAsk a question (type 'exit' to quit): ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    # Retrieve relevant chunks
    docs = db.similarity_search(question, k=3)

    if not docs:
        print("\nNo relevant information found in the PDF.")
        continue

    # Combine retrieved text
    context = "\n\n".join(doc.page_content for doc in docs)

    # Prompt
    prompt = f"""
You are a helpful PDF Question Answering assistant.

Use ONLY the information provided below.

If the answer is not available in the context, reply:
"The answer is not available in the provided PDF."

Give a clear and simple answer.

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = llm.invoke(prompt)

        print("\nAnswer:\n")

        # Print clean answer
        if hasattr(response, "content"):

            if isinstance(response.content, str):
                print(response.content)

            elif isinstance(response.content, list):
                for item in response.content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            print(item.get("text", ""))
                    else:
                        print(str(item))

            else:
                print(response.content)

        else:
            print(response)

    except Exception as e:
        print("Error:", e)