from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os

load_dotenv()

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="basic-rag-pipeline",
    embedding=embeddings_model
)

# Take user input 
user_query = input("Ask something about Node JS")

# Relevant chunks from the vector db
search_results =vector_db.similarity_search(query=user_query)

context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata["page_label"]}\nFile Location: {result.metadata["source"]}"
                         for result in search_results])

SYSTEM_PROMPT=f"""
You are helpful AI Assistant who answers user query with a quick overview based on the available context retrieved from a PDF
file along with page_contents and a page number.

You should only answer the user based on the following context and navigate the user to open the right page number to know more.
Context: 
{context}
""" 

openai_client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

response = openai_client.chat.completions.create(
    model="gemini-3-flash-preview",
    messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_query
        }
    ]
)

print(response.choices[0].message.content)