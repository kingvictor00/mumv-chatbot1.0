# To run this code, install the required libraries:
# pip install streamlit langchain langchain-community langchain-groq faiss-cpu sentence-transformers langchainhub youtube-transcript-api langchain-text-splitters langchain-huggingface langchain-classic beautifulsoup4 pytube python-dotenv

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # Loads .env locally or secrets on cloud

st.set_page_config(
    page_title="MumV",
    page_icon="🔹",
    layout="centered",
    initial_sidebar_state="collapsed"
)

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ==================== DARK-MODE STYLING ====================
st.markdown(
    """
    <style>
    .stApp {
        background: #0a0a0a;
        background-image: 
            linear-gradient(rgba(17,17,17,0.4) 1px, transparent 1px),
            linear-gradient(90deg, rgba(17,17,17,0.4) 1px, transparent 1px);
        background-size: 40px 40px;
        color: #e0e0e0;
    }
    header {visibility: hidden;}
    .st-emotion-cache-1y4p8pa {padding: 1rem 1rem 10rem 1rem !important;}
    .block-container {padding-top: 1rem !important; padding-bottom: 5rem !important;}
    .main-title {text-align: center; color: white; font-size: 3.8rem; font-weight: 800; margin: 0.4rem 0 0.1rem 0; font-family: 'Segoe UI', system-ui, sans-serif; letter-spacing: -1px;}
    .subtitle {text-align: center; color: #777; font-size: 0.95rem; font-weight: 500; margin: 0 0 2.5rem 0; letter-spacing: 1.5px; text-transform: uppercase;}
    .message-bubble {max-width: 78%; padding: 1rem 1.3rem; border-radius: 1.1rem; margin: 0.35rem 0; word-wrap: break-word; font-size: 1.03rem; line-height: 1.48;}
    .assistant .message-bubble {background: #1e1e1e; border: 1px solid #333; color: #f0f0f0; border-radius: 1.1rem 1.1rem 1.1rem 0.4rem; margin-right: auto; margin-left: 0.6rem;}
    .user .message-bubble {background: #2a3a4a; border: 1px solid #3a4a5a; color: #ffffff; border-radius: 1.1rem 1.1rem 0.4rem 1.1rem; margin-left: auto; margin-right: 0.6rem;}
    .message-label {font-size: 0.72rem; font-weight: 600; color: #666; margin-bottom: 0.35rem; letter-spacing: 0.8px; text-transform: uppercase;}
    .assistant .message-label {text-align: left; margin-left: 0.9rem;}
    .user .message-label {text-align: right; margin-right: 0.9rem;}
    .stChatInput > div > div > textarea {background-color: #111 !important; color: #ddd !important; border: 1px solid #333 !important; border-radius: 1.6rem !important; padding: 0.9rem 3.8rem 0.9rem 1.4rem !important; font-size: 1.05rem !important;}
    .stChatInput > div > div::after {content: "➤"; position: absolute; right: 1.1rem; top: 50%; transform: translateY(-50%); color: #888; font-size: 1.4rem; pointer-events: none;}
    .stChatInput textarea::placeholder {color: #777 !important;}
    </style>
    """,
    unsafe_allow_html=True
)

# Loading message
placeholder = st.empty()
placeholder.info("Loading knowledge base...")

# Sources (for reference only — loading happens in build_index.py)
sources = [
    "https://docs.getoptimum.xyz/",
    "https://docs.getoptimum.xyz/docs/learn/overview/deram",
    "https://docs.getoptimum.xyz/docs/learn/overview/p2p",
    "https://www.getoptimum.xyz/",
    "https://www.getoptimum.xyz/mump2p",
    "https://docs.getoptimum.xyz/docs/research/overview",
    "https://cryptorank.io/price/optimum"
]

# Title + subtitle
st.markdown('<div class="main-title">MUMV</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">BUILT BY VICTOR (@KINGINGVEEK)</div>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load vectorstore (from pre-built index)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
index_path = "faiss_index"

if "vectorstore" not in st.session_state:
    if os.path.exists(index_path):
        st.session_state.vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        st.error("Index not found. Run build_index.py locally and push the faiss_index/ folder.")

# Clear loading message
placeholder.empty()

# Debug sidebar
if "vectorstore" in st.session_state:
    doc_count = st.session_state.vectorstore.index.ntotal
    st.sidebar.success(f"✅ Knowledge base loaded — {doc_count} documents indexed")
    st.sidebar.info("Ask about Optimum founders, funding, mump2p, DeRAM, RLNC, Flexnodes, etc.")
else:
    st.sidebar.error("Failed to load knowledge base")

# LLM setup
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

# Improved prompt
template = """
You are MumV, a precise expert on Optimum (getoptimum.xyz).

Rules:
- Answer using ONLY the provided context.
- For questions about founders, funding, mump2p, DeRAM, RLNC, Flexnodes, or any Optimum detail → cite facts directly from context.
- If no relevant information in context → reply exactly: "I don't have that detail from the current Optimum sources. Can you clarify?"
- Never guess, assume, or use external knowledge for Optimum-specific questions.
- For general questions, answer normally.

Context:
{context}

Question: {question}

Answer:
"""
prompt_template = PromptTemplate.from_template(template)

# QA chain with improved retrieval
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=st.session_state.vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 15, "score_threshold": 0.3}
    ),
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt_template}
)

# ==================== CHAT RENDERING ====================
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    alignment_class = "user" if role == "user" else "assistant"
    label = "YOU" if role == "user" else "MUMV"

    with st.chat_message(role, avatar=None):
        st.markdown(
            f"""
            <div class="{alignment_class}">
                <div class="message-label">{label}</div>
                <div class="message-bubble">{content}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# User input
user_prompt = st.chat_input("Ask about Optimum or anything else...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    with st.chat_message("user", avatar=None):
        st.markdown(
            f"""
            <div class="user">
                <div class="message-label">YOU</div>
                <div class="message-bubble">{user_prompt}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with st.chat_message("assistant", avatar=None):
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"query": user_prompt})["result"]
        st.markdown(
            f"""
            <div class="assistant">
                <div class="message-label">MUMV</div>
                <div class="message-bubble">{response}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.session_state.messages.append({"role": "assistant", "content": response})
