# To run this code, install the required libraries:
# pip install streamlit langchain langchain-community langchain-groq faiss-cpu sentence-transformers langchainhub youtube-transcript-api langchain-text-splitters langchain-huggingface langchain-classic beautifulsoup4 pytube
# You need a Groq API key. Sign up at https://console.groq.com/ and set it as an environment variable: export GROQ_API_KEY="your_key"
# Then run: streamlit run this_file.py

import os
import streamlit as st
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
from dotenv import load_dotenv
load_dotenv()  # This loads variables from .env file
import time  # add this if not already imported

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
    .user .message-label     { text-align: right; margin-right: 0.9rem; }

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
placeholder.info("First-time setup: Loading Optimum knowledge base from official sources + downloading embedding model (~100MB). This can take 5–15 minutes on first run depending on your internet speed. Please keep this tab open and don't refresh.")

# Sources
sources = [
    "https://docs.getoptimum.xyz/",
    "https://www.getoptimum.xyz/",
    "https://x.com/get_optimum",
    "https://mirror.xyz/0xBfAC4db6d990A6cF9842f437345c447B18EbeF73",
    "https://cryptorank.io/price/optimum"
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
        # Load YouTube transcript
        youtube_loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
        docs.extend(youtube_loader.load())
        
        # Load web sources
        for url in sources:
            web_loader = WebBaseLoader(url)
            docs.extend(web_loader.load())
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(docs)
        
        # Create vectorstore
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(index_path)
        st.session_state.vectorstore = vectorstore

# Clear the loading message once vectorstore is ready
# Remove loading placeholder if it exists
if "placeholder" in locals():
    placeholder.empty()
# LLM setup
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)
# Custom prompt template
template = """
You are MumV, a helpful chatbot very knowledgeable about Optimum. 
Use the following context from official sources to answer the question accurately. 
If the information is not in the context, use your general knowledge to provide a helpful response, but do not hallucinate or make up facts.
If asked about your name, confirm you are MumV.

Context: {context}

Question: {question}

Answer:
"""
prompt_template = PromptTemplate.from_template(template)

# QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
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

# Chat input (placeholder customized in CSS)
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
