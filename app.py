# To run this code, install the required libraries:
# pip install streamlit langchain langchain-community langchain-groq faiss-cpu sentence-transformers langchainhub youtube-transcript-api langchain-text-splitters langchain-huggingface langchain-classic beautifulsoup4 pytube python-dotenv
# You need a Groq API key set in Streamlit secrets as GROQ_API_KEY

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env file (local) or secrets (cloud)

st.set_page_config(
    page_title="MumV",
    page_icon="🔹",
    layout="centered",
    initial_sidebar_state="collapsed"
)

from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# ==================== CUSTOM DARK-MODE STYLING ====================
st.markdown(
    """
    <style>
    /* Overall app background - pure dark with subtle grid */
    .stApp {
        background: #0a0a0a;
        background-image: 
            linear-gradient(rgba(17,17,17,0.4) 1px, transparent 1px),
            linear-gradient(90deg, rgba(17,17,17,0.4) 1px, transparent 1px);
        background-size: 40px 40px;
        color: #e0e0e0;
    }
    /* Hide default Streamlit header/footer/padding */
    header {visibility: hidden;}
    .st-emotion-cache-1y4p8pa {padding: 1rem 1rem 10rem 1rem !important;}
    .block-container {padding-top: 1rem !important; padding-bottom: 5rem !important;}
    /* Title styling */
    .main-title {
        text-align: center;
        color: white;
        font-size: 3.8rem;
        font-weight: 800;
        margin: 0.4rem 0 0.1rem 0;
        font-family: 'Segoe UI', system-ui, sans-serif;
        letter-spacing: -1px;
    }
    .subtitle {
        text-align: center;
        color: #777;
        font-size: 0.95rem;
        font-weight: 500;
        margin: 0 0 2.5rem 0;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    /* Chat message containers */
    .stChatMessage {
        background: transparent !important;
        padding: 0 !important;
        margin-bottom: 1.4rem !important;
    }
    /* Message bubbles */
    .message-bubble {
        max-width: 78%;
        padding: 1rem 1.3rem;
        border-radius: 1.1rem;
        margin: 0.35rem 0;
        position: relative;
        word-wrap: break-word;
        font-size: 1.03rem;
        line-height: 1.48;
    }
    /* MumV (assistant) messages - left side */
    .assistant .message-bubble {
        background: #1e1e1e;
        border: 1px solid #333;
        color: #f0f0f0;
        border-radius: 1.1rem 1.1rem 1.1rem 0.4rem;
        margin-right: auto;
        margin-left: 0.6rem;
    }
    /* User messages - right side */
    .user .message-bubble {
        background: #2a3a4a;
        border: 1px solid #3a4a5a;
        color: #ffffff;
        border-radius: 1.1rem 1.1rem 0.4rem 1.1rem;
        margin-left: auto;
        margin-right: 0.6rem;
    }
    /* Message labels */
    .message-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #666;
        margin-bottom: 0.35rem;
        letter-spacing: 0.8px;
        text-transform: uppercase;
    }
    .assistant .message-label { text-align: left; margin-left: 0.9rem; }
    .user .message-label { text-align: right; margin-right: 0.9rem; }
    /* Chat input bar */
    .stChatInput > div > div > textarea {
        background-color: #111 !important;
        color: #ddd !important;
        border: 1px solid #333 !important;
        border-radius: 1.6rem !important;
        padding: 0.9rem 3.8rem 0.9rem 1.4rem !important;
        font-size: 1.05rem !important;
    }
    .stChatInput > div > div::after {
        content: "➤";
        position: absolute;
        right: 1.1rem;
        top: 50%;
        transform: translateY(-50%);
        color: #888;
        font-size: 1.4rem;
        pointer-events: none;
    }
    /* Placeholder */
    .stChatInput textarea::placeholder {
        color: #777 !important;
    }
    /* Spinner / thinking indicator */
    .stSpinner > div > div {
        border-top-color: #4a90e2 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Show loading message right away
placeholder = st.empty()
placeholder.info("INITIALIZING... \nThis might take a while on first load")

# Sources (expanded for better coverage)
sources = [
    "https://docs.getoptimum.xyz/",                          # intro
    "https://docs.getoptimum.xyz/docs/learn/overview/deram", # DeRAM explanation
    "https://docs.getoptimum.xyz/docs/learn/overview/p2p",   # mump2p protocol
    "https://www.getoptimum.xyz/",                           # homepage
    "https://www.getoptimum.xyz/mump2p",                     # mump2p page
    "https://docs.getoptimum.xyz/docs/research/overview",    # technical papers
    "https://cryptorank.io/price/optimum"                    # price/funding
]
youtube_url = "https://www.youtube.com/watch?v=nLfegqPLY3o"

# Title + subtitle
st.markdown('<div class="main-title">MUMV</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">BUILT BY VICTOR (@KINGINGVEEK)</div>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load or create vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
index_path = "faiss_index"

if "vectorstore" not in st.session_state:
    if os.path.exists(index_path):
        st.session_state.vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        docs = []
        # Load YouTube transcript (safer mode)
        try:
            youtube_loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
            yt_docs = youtube_loader.load()
            if yt_docs:
                docs.extend(yt_docs)
                st.sidebar.info("YouTube transcript loaded successfully")
            else:
                st.sidebar.warning("YouTube: no transcript available")
        except Exception as e:
            st.sidebar.error(f"YouTube load failed: {e}")

        # Load web sources with robustness
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        for url in sources:
            try:
                loader = WebBaseLoader(
                    url,
                    header_template=headers,
                    autoset_encoding=True,
                    verify_ssl=True
                )
                page_docs = loader.load()
                if page_docs and len(page_docs[0].page_content.strip()) > 200:
                    docs.extend(page_docs)
                    st.sidebar.success(f"Loaded {url} ({len(page_docs[0].page_content)} chars)")
                else:
                    st.sidebar.warning(f"Empty content from {url}")
            except Exception as e:
                st.sidebar.error(f"Failed to load {url}: {str(e)}")

        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(docs)

        # Create vectorstore
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(index_path)
        st.session_state.vectorstore = vectorstore

# Clear loading message
placeholder.empty()

# Debug info in sidebar
if "vectorstore" in st.session_state:
    doc_count = st.session_state.vectorstore.index.ntotal
    st.sidebar.success(f"Vector store ready — {doc_count} documents indexed")
else:
    st.sidebar.error("Vector store failed to load")

# LLM setup
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

# Improved prompt (stricter on context)
template = """
You are MumV, a helpful and precise assistant deeply knowledgeable about Optimum (decentralized memory infrastructure from getoptimum.xyz).

Rules:
- ONLY use information present in the provided context.
- If the question is about Optimum, mump2p, DeRAM, Flexnodes, RLNC/gossip, or related topics, answer using context only.
- If no relevant information exists in context → say: "I don't have specific information about that from the official Optimum sources. Can you provide more details?"
- Never make up facts or use external knowledge for Optimum-specific questions.
- For general questions (not Optimum), you can answer normally.
- When possible, include key terms like mump2p, RLNC, Flexnodes.

Context from official sources:
{context}

Question: {question}

Answer:
"""
prompt_template = PromptTemplate.from_template(template)

# QA chain with better retrieval
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=st.session_state.vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 12, "score_threshold": 0.35}
    ),
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt_template}
)

# ==================== CHAT HISTORY RENDERING ====================
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    if role == "user":
        alignment_class = "user"
        label = "YOU"
    else:
        alignment_class = "assistant"
        label = "MUMV"

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

# Chat input
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
