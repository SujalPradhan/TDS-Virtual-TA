import json
import os
import traceback
from tqdm import tqdm
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def extract_content_from_files():
    """Extract content from markdown and JSON files"""
    print("📖 Extracting source material...")
    
    # Read markdown content as plain text
    try:
        with open("data/tds_content.json", encoding="utf-8") as file_handle:
            tools_content = file_handle.read()
    except Exception as error:
        print(f"Error reading markdown file: {error}")
        tools_content = ""
        
    # Parse forum discussions from JSON
    try:
        with open("data/discourse_forum_posts.json", encoding="utf-8") as file_handle:
            forum_discussions = json.load(file_handle)
    except Exception as error:
        print(f"Error reading forum data: {error}")
        forum_discussions = []
        
    return tools_content, forum_discussions

def segment_content_into_documents(tools_content, forum_discussions):
    """Transform raw content into document segments for processing"""
    content_documents = []
    
    # Create document from tools content
    tools_doc = Document(
        page_content=tools_content,
        metadata={
            "title": "Tools in Data Science",
            "source_url": "https://tds.s-anand.net"
        }
    )
    content_documents.append(tools_doc)
    
    # Process each forum post as a separate document
    for discussion in tqdm(forum_discussions, desc="Processing forum posts"):
        post_url = discussion.get("topic_url", "Source unavailable")
        post_title = discussion.get("title", "Untitled post")
        
        forum_doc = Document(
            page_content=discussion["content"],
            metadata={
                "title": post_title, 
                "source_url": post_url
            }
        )
        content_documents.append(forum_doc)
    
    # Divide documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    return text_splitter.split_documents(content_documents)

def generate_vector_embeddings(document_chunks):
    """Convert document chunks to vector embeddings"""
    try:
        print(f"🔢 Converting {len(document_chunks)} text segments to vector space...")
        # Use local embedding model to avoid API dependencies
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(document_chunks, embedding_model)
        return vector_db
    except Exception as error:
        print(f"Embedding generation failed: {error}")
        print(traceback.format_exc())
        raise RuntimeError("Vector embedding process failed. Check the logs for details.")

def persist_vector_database(vector_db, storage_dir="vectorstore"):
    """Save vector database to disk for later retrieval"""
    # Ensure storage directory exists
    os.makedirs(storage_dir, exist_ok=True)
    
    # Save the index to the specified path
    vector_db.save_local(storage_dir)
    print(f"💾 Vector database saved to {storage_dir}")

def process_and_index_content():
    """Main processing pipeline to convert content to searchable vector database"""
    print("🚀 Starting content processing pipeline...")
    
    # Step 1: Extract content from source files
    tools_content, forum_discussions = extract_content_from_files()
    
    # Step 2: Transform content into searchable document chunks
    document_chunks = segment_content_into_documents(tools_content, forum_discussions)
    
    # Step 3: Generate vector embeddings from document chunks
    vector_db = generate_vector_embeddings(document_chunks)
    
    # Step 4: Save vector database to disk
    persist_vector_database(vector_db)
    
    print("✅ Content processing complete!")

if __name__ == "__main__":
    process_and_index_content()