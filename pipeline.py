"""
Knowledge Retrieval and Response Generation System
This module handles vector storage operations and AI-powered question answering
"""

import json
import os
import traceback
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv
from pathlib import Path

# Make sure we're loading from the correct path
dotenv_path = Path(os.path.dirname(os.path.abspath(__file__))) / '.env'
load_dotenv(dotenv_path=dotenv_path)

import httpx
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings


# Configuration for API access
API_KEY = os.environ.get("AIPROXY_TOKEN")
API_ENDPOINT = "https://aiproxy.sanand.workers.dev/openai"
print("🔑 API key verification status:", "Available" if API_KEY else "Missing")
if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please check your .env file.")
print(f"API key length: {len(API_KEY) if API_KEY else 0}")

def create_text_embeddings(text_fragments: List[str]) -> List[List[float]]:
    """
    Transform text into numerical vector representations using embedding API
    
    Args:
        text_fragments: List of text strings to convert to embeddings
        
    Returns:
        List of embedding vectors (each vector is a list of floats)
    """
    # Use HuggingFace embeddings for consistency with data preprocessing
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return [embedding_model.embed_query(text) for text in text_fragments]

def initialize_vector_database(storage_location: str = "vectorstore") -> Any:
    """
    Load vector database from disk storage
    
    Args:
        storage_location: Path to the stored vector database
        
    Returns:
        Initialized vector database instance
    """
    try:
        # Configure embeddings with the same model used for creating the database
        embeddings_config = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load the vector database from disk
        return FAISS.load_local(
            storage_location, 
            embeddings_config, 
            allow_dangerous_deserialization=True
        )
    except Exception as error:
        print(f"Failed to initialize vector database: {error}")
        print(traceback.format_exc())
        raise

def retrieve_and_generate_answer(query: str, knowledge_base: Any, result_count: int = 5) -> Tuple[str, List[Dict[str, str]]]:
    """
    Process a question by retrieving relevant context and generating an answer
    
    Args:
        query: The user's question
        knowledge_base: Vector database containing knowledge
        result_count: Number of relevant documents to retrieve
        
    Returns:
        Tuple of (answer_text, source_references)
    """
    try:
        # Convert question to vector representation
        query_vector = create_text_embeddings([query])[0]
        
        # Find relevant documents in the knowledge base
        relevant_docs = knowledge_base.similarity_search_by_vector(
            query_vector, 
            k=result_count
        )
        
        # Combine retrieved documents into context
        knowledge_context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
        # Prepare request to language model
        request_endpoint = f"{API_ENDPOINT}/v1/chat/completions"
        auth_headers = {"Authorization": f"Bearer {API_KEY}"}
        
        # Structure request body with system message and user query
        request_body = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an educational assistant for the Tools for Data Science course. "
                        "Provide clear, accurate responses based on this reference material:\n\n"
                        f"{knowledge_context}\n\n"
                        "If source information is available, format your response as JSON with this structure:\n"
                        "{\n  \"answer\": \"your detailed response\",\n  \"links\": [\n    {\"url\": \"source_url\", \"text\": \"brief description\"}, ...\n  ]\n}\n"
                        "If no source information is available, provide a plain text response."
                    )
                },
                {"role": "user", "content": query}
            ]
        }

        # Send request to language model
        try:
            with httpx.Client(timeout=60.0) as http_client:  # Increased timeout
                response = http_client.post(
                    request_endpoint, 
                    headers=auth_headers, 
                    json=request_body
                )
                response.raise_for_status()
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as error:
            print(f"HTTP error during answer generation: {error.response.status_code}, {error.response.text}")
            raise
        except httpx.RequestError as error:
            print(f"Request error during answer generation: {error}")
            raise

        # Parse the response, handling both JSON and plain text formats
        try:
            parsed_response = json.loads(generated_text)

            answer_content = parsed_response.get("answer", generated_text)

            source_links = parsed_response.get("links", [])



            # If LLM gave empty links, use fallback from relevant_docs

            if not source_links:

                for doc in relevant_docs:

                    source_url = doc.metadata.get("source_url", "Unknown source")

                    content_preview = doc.page_content.strip().split("\n")[0][:300]

                    source_links.append({

                        "url": source_url,

                        "text": content_preview

                    })
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"JSON parsing error: {e}, falling back to plain text")
            # Fall back to plain text response with automatic source references
            answer_content = generated_text
            source_links = []
            
            # Extract source information from retrieved documents
            for doc in relevant_docs:
                source_url = doc.metadata.get("source_url", "Unknown source")
                # Use first line or fragment of content as description
                content_preview = doc.page_content.strip().split("\n")[0][:300]
                source_links.append({
                    "url": source_url,
                    "text": content_preview
                })

        return answer_content, source_links
        
    except Exception as error:
        print(f"Error in retrieve_and_generate_answer: {error}")
        print(traceback.format_exc())
        raise