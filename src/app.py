import streamlit as streamlit
from langchain_core.messages import AIMessage, HumanMessage

from response_generator import generate_response
from retriever import get_retriever_chain
from embeddings import ollama_emb
from vector_store import get_vector_store_from_url

# app configuration
streamlit.set_page_config(page_title="Chat with Websites", page_icon=" 🤖")
streamlit.title("Chat with Websites")

# sidebar
with streamlit.sidebar:
    streamlit.header("Settings")
    website_url = streamlit.text_input("URL", "")

if website_url is None or website_url == "":
    streamlit.info("Please enter a website URL")
    streamlit.stop()

else:
    # create state
    if "chat_history" not in streamlit.session_state:
        streamlit.session_state.chat_history = [AIMessage(content="Hi, I'm an AI bot. How can I help you?")]

    if "vector_store" not in streamlit.session_state:
        streamlit.session_state.vector_store = get_vector_store_from_url(website_url, ollama_emb)

    # user input
    user_query = streamlit.chat_input("Type your message...")
    if user_query is not None and user_query != "":

        retriever_chain = get_retriever_chain(streamlit.session_state.vector_store)
        response = generate_response(user_query, retriever_chain, streamlit.session_state.chat_history)
        streamlit.session_state.chat_history.append(HumanMessage(content=user_query))
        streamlit.session_state.chat_history.append(AIMessage(content=response))

    for message in streamlit.session_state.chat_history:
        if isinstance(message, AIMessage):
            with streamlit.chat_message("AI"):
                streamlit.write(message.content)
        elif isinstance(message, HumanMessage):
            with streamlit.chat_message("Human"):
                streamlit.write(message.content)
