
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from llm import llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Load document
loader = TextLoader("Speech.txt")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embeddings
emb = OllamaEmbeddings(model="qwen2:0.5b")

# Vector DB
db = FAISS.from_documents(chunks, emb)
retriever = db.as_retriever()

llm = Ollama(model="qwen2:0.5b")

prompt = ChatPromptTemplate.from_template("""
Use Only the following context to answer the question:

Context:
{context}

Question:
{question}

Answer briefly:
""")

# Build LCEL RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Query
response = rag_chain.invoke(input("Ask anything from the Document"))

print("\n📌 RAG Output:\n", response)