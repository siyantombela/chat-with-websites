from langchain.chains import create_history_aware_retriever
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from models import llm


def get_retriever_chain(vector_store: Chroma) -> callable:
    """
    Create a retriever chain from a vector store
    """
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query in order to get information relevant to this"
         " conversation")
    ])
    retriever = vector_store.as_retriever()
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain
