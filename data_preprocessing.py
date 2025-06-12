from dotenv import load_dotenv
load_dotenv()

import json
import os
from tqdm import tqdm
import numpy as np
from collections import namedtuple
import pickle
import shutil

# Custom document structure
TextDocument = namedtuple('TextDocument', ['content', 'metadata'])

# Set up similarity search framework
class TextVectorIndex:
    def __init__(self, embedding_func):
        self.embedding_function = embedding_func
        self.vectors = []
        self.documents = []
        
    def add_documents(self, documents):
        """Add documents to the vector index"""
        print("Generating vector embeddings...")
        for doc in tqdm(documents):
            vector = self.embedding_function([doc.content])[0]
            self.vectors.append(vector)
            self.documents.append(doc)
            
    def save(self, directory):
        """Save the index to disk"""
        os.makedirs(directory, exist_ok=True)
        # Create a binary vector file using numpy
        vector_array = np.array(self.vectors, dtype=np.float32)
        np.save(os.path.join(directory, "index.npy"), vector_array)
        
        # Save documents separately
        with open(os.path.join(directory, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
            
        # Create a faiss-compatible index file for compatibility
        try:
            import faiss
            dimension = len(self.vectors[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(vector_array)
            faiss.write_index(index, os.path.join(directory, "index.faiss"))
            
            # Create compatibility pkl file
            shutil.copy(os.path.join(directory, "documents.pkl"), 
                       os.path.join(directory, "index.pkl"))
        except ImportError:
            print("Warning: faiss not available, creating only numpy storage")

# Import needed for API compatibility
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document 
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load data from source files
def load_data():
    with open("data/tds_content.json", encoding="utf-8") as f:
        tds_data = json.load(f)
    with open("data/discourse_forum_posts.json", encoding="utf-8") as f:
        discourse_data = json.load(f)
    return tds_data, discourse_data

# Process content into manageable text segments
def chunk_data(tds_data, discourse_data):
    documents = []

    # Process TDS content
    for topic in tqdm(tds_data, desc="TDS"):
        url = topic.get("url", "https://tds.s-anand.net")
        document = Document(
            page_content=topic["content"],
            metadata={"title": topic["title"], "source": url}
        )
        documents.append(document)

    # Process discourse forum data
    for post in tqdm(discourse_data, desc="Discourse"):
        source = post.get("topic_url", "Unknown")
        title = post.get("title", source)
        document = Document(
            page_content=post["content"],
            metadata={"title": title, "source": f"Discourse: {source}"}
        )
        documents.append(document)

    # Split into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# Create vector embeddings from text
def embed_text(docs):
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("AIPROXY_TOKEN"),
        base_url="https://aiproxy.sanand.workers.dev/openai/v1"
    )
    return FAISS.from_documents(docs, embedding_model)

# Store the vector index
def save_index(index, path="vectorstore"):
    os.makedirs(path, exist_ok=True)
    index.save_local(path)

# Main process to index all content
def index_data():
    print("🔍 Processing and segmenting content...")
    tds_data, discourse_data = load_data()
    text_segments = chunk_data(tds_data, discourse_data)

    print(f"✍️ Creating embeddings for {len(text_segments)} text segments...")
    vector_index = embed_text(text_segments)

    print("💾 Persisting vector database...")
    save_index(vector_index)

    print("✅ Indexing process completed!")

if __name__ == "__main__":
    index_data()