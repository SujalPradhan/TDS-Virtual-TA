"""
TDS Knowledge Retrieval System
This module manages semantic search operations and LLM-assisted question answering capabilities
"""

import json
from dotenv import load_dotenv
load_dotenv()

import os
import numpy as np
import httpx
import time
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

# === Service access configuration ===
API_TOKEN = os.environ.get("AIPROXY_TOKEN")
API_ENDPOINT = "https://aiproxy.sanand.workers.dev/openai"
print("🔑 API token status:", "Available" if API_TOKEN else "Missing")

# === Vector encoding service ===
def generate_embedding(text_fragments):
    """Transform text into numerical vectors"""
    endpoint = f"{API_ENDPOINT}/v1/embeddings"
    auth_header = {"Authorization": f"Bearer {API_TOKEN}"}
    request_body = {
        "model": "text-embedding-3-small",
        "input": text_fragments
    }

    # Send request with timeout protection
    with httpx.Client(timeout=30.0) as session:
        try:
            api_response = session.post(endpoint, headers=auth_header, json=request_body)
            api_response.raise_for_status()
            response_data = api_response.json()
            return [record["embedding"] for record in response_data["data"]]
        except (httpx.RequestError, httpx.HTTPStatusError) as err:
            print(f"Error generating embeddings: {err}")
            raise

# === Vector database access ===
# def initialize_knowledge_base(directory_path="vectorstore"):
#     """Load vector database from disk"""
#     # Create wrapper for compatibility with stored indexes
#     embedding_interface = OpenAIEmbeddings(
#         model="text-embedding-3-small",
#         api_key="sk-placeholder",
#         base_url="https://aiproxy.sanand.workers.dev/openai/v1"
#     )
#     return FAISS.load_local(directory_path, embedding_interface, allow_dangerous_deserialization=True)

from langchain_community.document_loaders import TextLoader

def initialize_knowledge_base(directory_path="vectorstore"):
    """Load or build vector database"""
    embedding_interface = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key="sk-placeholder",
        base_url="https://aiproxy.sanand.workers.dev/openai/v1"
    )

    index_file = os.path.join(directory_path, "index.faiss")
    
    if os.path.exists(index_file):
        print("✅ Found FAISS index. Loading from disk...")
        return FAISS.load_local(directory_path, embedding_interface, allow_dangerous_deserialization=True)
    else:
        print("⚠️ FAISS index not found. Rebuilding knowledge base...")
        # Load documents – replace this with your actual data loading logic
        loader = TextLoader("data/tds_content.txt")  # or load PDF, web data, etc.
        documents = loader.load()
        
        # Build vectorstore and save it
        knowledge_base = FAISS.from_documents(documents, embedding_interface)
        knowledge_base.save_local(directory_path)
        print("✅ Vectorstore rebuilt and saved.")
        return knowledge_base

# === Query processing system ===
def process_query(user_query, knowledge_base, result_count=5):
    """Find relevant information and generate response"""
    # Convert question to vector representation
    start_time = time.time()
    query_vector = np.array(generate_embedding([user_query])[0], dtype=np.float32)
    
    # Ensure dimensional compatibility with knowledge base
    vector_dimension = knowledge_base.index.d
    
    # Handle dimension mismatch if necessary
    if len(query_vector) != vector_dimension:
        print(f"⚠️ Vector dimension mismatch. Got {len(query_vector)}, expected {vector_dimension}")
        # Align dimensions through padding or truncation
        if len(query_vector) < vector_dimension:
            query_vector = np.pad(query_vector, (0, vector_dimension - len(query_vector)))
        else:
            query_vector = query_vector[:vector_dimension]
    
    # Retrieve most relevant documents
    relevant_docs = knowledge_base.similarity_search_by_vector(query_vector, k=result_count)
    relevant_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    # Prepare for LLM consultation
    endpoint = f"{API_ENDPOINT}/v1/chat/completions"
    auth_header = {"Authorization": f"Bearer {API_TOKEN}"}
    request_body = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable Teaching Assistant for the Tools for Data Science (TDS) course. "
                    "Respond to questions based on this reference information:\n\n"
                    f"{relevant_text}\n\n"
                    "If reference information includes source links, format your response as JSON:\n"
                    "{\n  \"answer\": \"...\",\n  \"links\": [\n    {\"url\": \"...\", \"text\": \"...\"}, ...\n  ]\n}\n"
                    "Otherwise, provide a plain text response."
                )
            },
            {"role": "user", "content": user_query}
        ]
    }

    # Request answer from language model
    with httpx.Client(timeout=30.0) as session:
        api_response = session.post(endpoint, headers=auth_header, json=request_body)
        api_response.raise_for_status()
        response_data = api_response.json()
        model_response = response_data["choices"][0]["message"]["content"].strip()

    # Process and structure the response
    try:
        parsed_response = json.loads(model_response)
        final_answer = parsed_response["answer"]
        reference_links = parsed_response.get("links", [])
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fall back to plain text with auto-extracted references
        final_answer = model_response
        reference_links = []
        for doc in relevant_docs:
            source_url = doc.metadata.get("source", "Unknown")
            content_preview = doc.page_content.strip().split("\n")[0][:300]
            reference_links.append({
                "url": source_url,
                "text": content_preview
            })

    print(f"⏱️ Query processed in {time.time() - start_time:.2f} seconds")
    return final_answer, reference_links