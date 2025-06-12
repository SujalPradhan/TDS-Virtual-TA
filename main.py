import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from fastapi.responses import JSONResponse
import logging
import time
import traceback
from contextlib import asynccontextmanager
from pipeline import initialize_knowledge_base, process_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tds-virtual-ta")

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load knowledge base once
    logger.info("🔄 Initializing TDS Virtual Teaching Assistant...")
    app.state.knowledge_base = initialize_knowledge_base()
    logger.info("✅ Knowledge base loaded successfully!")
    yield
    # Shutdown: Cleanup
    logger.info("👋 Shutting down TDS Virtual Teaching Assistant...")

# Initialize FastAPI application
app = FastAPI(
    title="TDS Virtual Teaching Assistant",
    description="API for answering questions about Tools for Data Science course",
    version="1.0.0",
    lifespan=lifespan
)

# Configure Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response data models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask the TDS Virtual TA")
    image: Optional[str] = Field(None, description="Optional base64-encoded image for future visual context support")

class LinkItem(BaseModel):
    url: str = Field(..., description="Source URL for the reference")
    text: str = Field(..., description="Preview text from the source")

class ExactResponseFormat(BaseModel):
    answer: str = Field(..., description="The response to the user's question")
    links: List[LinkItem] = Field(default_factory=list, description="References used to generate the answer")

@app.post("/query", response_model=ExactResponseFormat, tags=["Query"])
async def handle_query(data: QueryRequest):
    """Process a question and generate a response using the TDS knowledge base"""
    start_time = time.time()
    try:
        if not os.environ.get("AIPROXY_TOKEN"):
            logger.error("🔑 API token missing. Please check environment configuration.")
            raise HTTPException(
                status_code=503,
                detail="API authentication token not configured. Please contact the administrator."
            )

        # Process the question
        answer_text, reference_sources = process_query(
            data.question, 
            app.state.knowledge_base
        )

        # Return exactly the required format
        response_data = ExactResponseFormat(
            answer=answer_text,
            links=[LinkItem(**source) for source in reference_sources]
        )
        
        logger.info(f"✅ Query processed in {time.time() - start_time:.2f} seconds")
        return response_data

    except Exception as error:
        error_details = traceback.format_exc()
        logger.error(f"❌ Error processing query: {str(error)}")
        logger.debug(f"Detailed error: {error_details}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": str(error), 
                "details": error_details.split("\n")
            }
        )

# For backward compatibility - old route that also returns in the exact format
@app.post("/", response_model=ExactResponseFormat)
async def legacy_handle_query(data: QueryRequest):
    """Legacy endpoint that redirects to the main query handler"""
    return await handle_query(data)

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """Check system health and readiness"""
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "api_token_configured": bool(os.environ.get("AIPROXY_TOKEN"))
    }

# Development server configuration
# if __name__ == "__main__":
#     import uvicorn
#     logger.info("🚀 Starting development server...")
#     uvicorn.run(
#         app, 
#         host="0.0.0.0", 
#         port=8000,
#         log_level="info"
#     )