import streamlit
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings


def get_vector_store_from_url(url: str, embeddings: OllamaEmbeddings) -> Chroma:
    """
    Create a vector store from a website url
    """
    streamlit.write("Chatting with context from source: " + url)
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    document_chunks = text_splitter.split_documents(document)
    vector_store = Chroma.from_documents(document_chunks, embeddings)
    return vector_store
