import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse
from pipeline import initialize_vector_database, create_text_embeddings, retrieve_and_generate_answer
import traceback  # Added for detailed error logging

# Initialize FastAPI application
app = FastAPI()

# Configure Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the vector database once during application startup
knowledge_base = initialize_vector_database()

# Define request and response data models
class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64-encoded string for future image support

class SourceReference(BaseModel):
    url: str
    text: str

class QuestionResponse(BaseModel):
    answer: str
    links: List[SourceReference]

@app.post("/", response_model=QuestionResponse)
async def process_question(data: QuestionRequest):
    """
    Process a user question and return an AI-generated answer with source references
    """
    try:
        # Verify API key is available
        if not os.environ.get("AIPROXY_TOKEN"):
            return JSONResponse(
                status_code=500,
                content={"error": "AIPROXY_TOKEN environment variable is not set. Please check your .env file."}
            )
            
        # Retrieve answer and sources using the knowledge retrieval system
        answer_text, source_references = retrieve_and_generate_answer(
            data.question, 
            knowledge_base
        )

        # Return structured response
        return JSONResponse(
            content={
                "answer": answer_text,
                "links": source_references
            },
            media_type="application/json"
        )

    except Exception as error:
        # Enhanced error handling with traceback
        error_traceback = traceback.format_exc()
        print(f"Error processing question: {str(error)}")
        print(f"Traceback: {error_traceback}")
        
        return JSONResponse(
            status_code=500,
            content={"error": str(error), "traceback": error_traceback.split("\n")}
        )

# Development server configuration
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         app, 
#         host="0.0.0.0", 
#         port=8000,  # Changed from 8000 to 8001
#         log_level="info"
#     )