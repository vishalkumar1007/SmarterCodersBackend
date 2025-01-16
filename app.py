from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils.htmlParser import fetch_html_content, parse_html_content
from utils.tokenizer import tokenize_and_embed
from milvusClient import MilvusClient
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize Milvus client
try:
    milvus_client = MilvusClient()
    print("Milvus client initialized successfully")
except Exception as e:
    print(f"Failed to initialize Milvus client: {e}")

# Request body model
class SearchRequest(BaseModel):
    url: str
    query: str

@app.post("/search")
async def search(request: SearchRequest):
    try:
        print(f"Received search request for URL: {request.url}")
        
        # Fetch and clean HTML content
        html_content = fetch_html_content(request.url)
        cleaned_content = parse_html_content(html_content)
        
        # Tokenize and embed content
        chunks, embeddings = tokenize_and_embed(cleaned_content)
        
        print(f"Processing {len(chunks)} chunks of content")
        
        # Index chunks in Milvus
        milvus_client.index_data(chunks, embeddings)
        
        # Perform search
        results = milvus_client.search(request.query)
        
        # Return formatted results
        response = {
            "results": results,
            "metadata": {
                "total_chunks": len(chunks),
                "chunks_searched": min(10, len(chunks)),
                "query": request.query
            }
        }
        
        return response

    except Exception as e:
        print(f"Error in search endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))