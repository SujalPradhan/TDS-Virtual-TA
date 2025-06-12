"""
Script to test API key configuration and connection
"""
import os
import httpx
from dotenv import load_dotenv
load_dotenv()

# Check if API key is set
api_key = os.environ.get("AIPROXY_TOKEN")
print(f"API key exists: {api_key is not None}")
print(f"API key length: {len(api_key) if api_key else 0}")

# Test a simple API call
if api_key:
    try:
        print("Testing API connection...")
        
        API_ENDPOINT = "https://aiproxy.sanand.workers.dev/openai"
        request_endpoint = f"{API_ENDPOINT}/v1/embeddings"
        auth_headers = {"Authorization": f"Bearer {api_key}"}
        request_body = {
            "model": "text-embedding-3-small",
            "input": ["Test connection"]
        }
        
        with httpx.Client(timeout=30.0) as http_client:
            response = http_client.post(
                request_endpoint, 
                headers=auth_headers, 
                json=request_body
            )
            response.raise_for_status()
            result_data = response.json()
            print("API connection successful!")
            print(f"Response status: {response.status_code}")
            print(f"First few embedding values: {result_data['data'][0]['embedding'][:5]}")
    except Exception as e:
        print(f"API connection failed: {e}")
else:
    print("Cannot test connection without API key")