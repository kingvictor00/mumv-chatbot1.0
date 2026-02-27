import os
from langchain_community.document_loaders import PlaywrightURLLoader, WebBaseLoader, YoutubeLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
os.environ["USER_AGENT"] = "MumV-Chatbot/1.0 (personal project; contact: @kingingveek)"

print("Starting offline index build...")

sources = [
    "https://docs.getoptimum.xyz/",
    "https://docs.getoptimum.xyz/docs/learn/overview/deram",
    "https://docs.getoptimum.xyz/docs/learn/overview/p2p",
    "https://www.getoptimum.xyz/",
    "https://www.getoptimum.xyz/mump2p",
    "https://docs.getoptimum.xyz/docs/research/overview",
    "https://cryptorank.io/price/optimum"
]
youtube_url = "https://www.youtube.com/watch?v=nLfegqPLY3o"

docs = []

# YouTube transcript
try:
    yt_loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
    yt_docs = yt_loader.load()
    if yt_docs:
        docs.extend(yt_docs)
        print("YouTube transcript loaded successfully")
    else:
        print("YouTube: no transcript content")
except Exception as e:
    print(f"YouTube failed: {e}")

# Dynamic pages (use Playwright)
js_urls = [
    "https://www.getoptimum.xyz/",
    "https://www.getoptimum.xyz/mump2p",
    "https://docs.getoptimum.xyz/",
    "https://docs.getoptimum.xyz/docs/learn/overview/deram",
    "https://docs.getoptimum.xyz/docs/learn/overview/p2p",
]

static_urls = [
    "https://docs.getoptimum.xyz/docs/research/overview",
    "https://cryptorank.io/price/optimum"
]

print("Loading dynamic pages with Playwright...")
try:
    pw_loader = PlaywrightURLLoader(
        urls=js_urls,
        remove_selectors=["header", "footer", "nav", "script", "style", "iframe", "noscript"],
        headless=True,
        timeout=60000  # 60s per page
    )
    pw_docs = pw_loader.load()
    loaded = 0
    for doc in pw_docs:
        content = doc.page_content.strip()
        if len(content) > 300:
            docs.append(doc)
            loaded += 1
    print(f"Playwright loaded {loaded}/{len(js_urls)} dynamic pages successfully")
except Exception as e:
    print(f"Playwright failed (falling back): {e}")

# Static pages fallback
print("Loading static pages...")
for url in static_urls + js_urls:  # retry dynamic ones if Playwright failed
    try:
        loader = WebBaseLoader(url)
        page_docs = loader.load()
        content = page_docs[0].page_content.strip() if page_docs else ""
        if len(content) > 200:
            docs.extend(page_docs)
            print(f"Loaded via WebBaseLoader: {url} ({len(content)} chars)")
        else:
            print(f"Empty from {url}")
    except Exception as e:
        print(f"Failed {url}: {e}")

if not docs:
    print("WARNING: No documents loaded!")
else:
    print(f"Total documents collected: {len(docs)}")

# Split & embed
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)
vectorstore.save_local("faiss_index")

print("Index built and saved to faiss_index/. You can now run the Streamlit app.")
