import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os

# Load secrets
load_dotenv()
st.set_page_config(page_title="ChatGPT Clone", layout="centered")
st.title("🗨️ JeetGPT — Your AI Chatbot")

# Save conversation history in Streamlit session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.chat_input("Ask me anything...")

# Set up model
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    temperature=0.3,
    max_new_tokens=512
)
model = ChatHuggingFace(llm=llm)

# Handle input
if user_input:
    # Display user message
    st.session_state.chat_history.append(("user", user_input))
    with st.spinner("Thinking..."):
        response = model.invoke(user_input)
        st.session_state.chat_history.append(("bot", response.content))

# Render chat history like bubbles
for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.chat_message("user").markdown(msg)
    else:
        st.chat_message("assistant").markdown(msg)
