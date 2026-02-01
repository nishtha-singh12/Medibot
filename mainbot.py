import os
import streamlit as st
from PIL import Image

# 👁️ Import your image retrieval pipeline
from main_rag_pipeline import get_best_image_from_query

# 🧠 Imports for Ollama RAG
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama

# Global constants
DB_FAISS_PATH = "vectorstore/db_faiss"

# 🗄️ Load vectorstore with caching
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# 🪄 Custom prompt
def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# 🚀 Load Ollama LLM
def load_llm():
    llm = Ollama(
        model="qwen3:4b",
        temperature=0.5
    )
    return llm

# 🧠 Ollama RAG Chatbot Page
def rag_chatbot_page():
    st.title("🧠 Ask Medical Chatbot (Local RAG with Ollama)")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Ask your medical question...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
You are a highly accurate and concise medical assistant chatbot.

Use the following context and your knowledge to answer the user’s medical question truthfully and concisely.

Caution: Only answer if you're confident. If unsure or the question requires a licensed professional, say:
*"I'm not qualified to give a definitive answer. Please consult a licensed healthcare provider."*

Context: {context}
Question: {question}

Answer:
"""

        try:
            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            result_to_show = result + "\n\n**Source Documents:**\n" + str(source_documents)
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")

# 👁️ Eye Image Generator Page
def eye_image_generator_page():
    st.title("👁️ Semantic Eye Image Generator")

    user_input = st.text_input("Ask about any eye part or structure to get a matched image:")

    if user_input:
        with st.spinner("Searching for the best matching image..."):
            image_path = get_best_image_from_query(user_input)
        
        if image_path:
            st.image(image_path, caption="Matched Image", use_container_width=True)
        else:
            st.error("No image matched your query.")

# 🚦 Main App Selector
def main():
    st.set_page_config(page_title="Medical RAG + Image Bot", layout="centered")

    st.sidebar.title("Navigation")
    choice = st.sidebar.radio(
        "Select Mode:",
        ["🧠 Chat with RAG Bot", "👁️ Generate Eye Image"]
    )

    if choice == "🧠 Chat with RAG Bot":
        rag_chatbot_page()
    elif choice == "👁️ Generate Eye Image":
        eye_image_generator_page()

if __name__ == "__main__":
    main()
