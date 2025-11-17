import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os

st.set_page_config(page_title="Local RAG Chatbot", layout="wide")

st.title("📘 Local RAG Chatbot (Ollama + LangChain 0.3)")

# Upload document
uploaded_file = st.file_uploader("Upload a .txt document", type=["txt"])

if uploaded_file:
    data_path = r"C:\Users\dell\Desktop\cc\Speech.txt"
    with open(data_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("File uploaded successfully!")

    # Load document
    loader = TextLoader(data_path)
    docs = loader.load()

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embeddings
    emb = OllamaEmbeddings(model="qwen2:0.5b")

    # Create FAISS vector DB
    db = FAISS.from_documents(chunks, emb)
    retriever = db.as_retriever()

    # LLM
    llm = Ollama(model="qwen2:0.5b")

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    Context:
    {context}

    Question:
    {question}

    Answer concisely:
    """)

    # RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # Chat section
    st.subheader("Ask Questions")

    question = st.text_input("Enter your question")

    if st.button("Submit") and question:
        answer = rag_chain.invoke(question)
        st.write("### Answer:")
        st.write(answer)
