from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path = Path(__file__).parent / "nodejs.pdf"

# Load this file in python program 
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()

# Split the docs into smaller chunks 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(docs)

print(chunks)

#Vector Embeddings
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="basic-rag-pipeline"
)

print("Indexing of documents done....")