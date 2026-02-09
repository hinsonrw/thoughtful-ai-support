"""
Streamlit interface for Thoughtful AI support agent
"""

import streamlit as st
from agent import ThoughtfulAgent

st.set_page_config(page_title="Thoughtful AI Support")

st.title("Thoughtful AI Customer Support")

@st.cache_resource
def get_agent():
    return ThoughtfulAgent()

# Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    agent = get_agent()
except Exception as e:
    st.error(f"Failed to initialize: {e}")
    st.info("Make sure OPENAI_API_KEY is set in your .env file")
    st.stop()

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.text(f"You: {msg['content']}")
    else:
        st.text(f"openai: {msg['content']}")
    st.text("")

# Input
prompt = st.chat_input("Ask a question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    response, source = agent.respond(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})

    st.rerun()
