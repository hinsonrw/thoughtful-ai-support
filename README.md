# Thoughtful AI Support Agent

This is a simple customer support chatbot for Thoughtful AI built with Streamlit and OpenAI. It uses embeddings to match user questions against a set of predefined answers about Thoughtful AI products (EVA, CAM, PHIL). If no good match is found it falls back to GPT-4o-mini to generate a response.

To run it, install the requirements with pip install -r requirements.txt, add your OpenAI API key to the .env file, then run streamlit run app.py
