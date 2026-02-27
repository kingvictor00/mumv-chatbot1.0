import os
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()  # Loads .env file automatically
import os
os.environ["USER_AGENT"] = "MumV-Chatbot/1.0 (personal project)"

print("Starting offline index build...")

sources = [
    "https://docs.getoptimum.xyz/",
    "https://www.getoptimum.xyz/",
    "https://x.com/get_optimum",
    "https://mirror.xyz/0xBfAC4db6d990A6cF9842f437345c447B18EbeF73",
    "https://cryptorank.io/price/optimum"
]
youtube_url = "https://www.youtube.com/watch?v=nLfegqPLY3o"

docs = []
youtube_loader = YoutubeLoader.from_youtube_url(youtube_url)  # defaults to add_video_info=False
docs.extend(youtube_loader.load())

for url in sources:
    try:
        web_loader = WebBaseLoader(url)
        docs.extend(web_loader.load())
        print(f"Loaded: {url}")
    except Exception as e:
        print(f"Failed to load {url}: {e}")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local("faiss_index")

print("Index built and saved to faiss_index/ folder. You can now run the Streamlit app.")
