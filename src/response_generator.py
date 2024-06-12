import streamlit
from dotenv import load_dotenv

from conversational_chain import get_conversational_rag_chain
load_dotenv()


def generate_response(user_input: str, retriever_chain: callable, chat_history: list) -> str:
    """
    Generate a response to a user input
    """
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)
    ai_response = conversational_rag_chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })

    return ai_response['answer']
