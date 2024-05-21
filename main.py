from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from magicRAG import MagicRAG
import uvicorn
import os #core python module to allow env files to be loaded up
from dotenv import load_dotenv, dotenv_values

# Define a Pydantic model to structure the JSON request for queries
class Query(BaseModel):
    question: str

app = FastAPI()

@app.get("/api/home")
async def return_home():
    # Placeholder for where you might interact with MagicRAG
    # For example, if you need to initialize or fetch something on app start.
    llm_response = "Initial placeholder response"
    return {
        "message": "Hello World from the FastAPI server!",
        "llm_response": llm_response
    }

@app.post("/api/query")
async def handle_query(query: Query):
    try:
        myRAG = MagicRAG(og_user_query=query.question)
        llm_final_output = myRAG.output()
        return {
            "message": "Response from the FastAPI server's handle_query route!",
            "llm_response": llm_final_output
        }
    except Exception as e:
        # You can customize the error handling here
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Use uvicorn to run the app; by default it will be on http://127.0.0.1:8000
    uvicorn.run(app, host="127.0.0.1", port=8080)

@app.get("/")
async def root():
    return {"message": "Magic RAG LLM Judge API v0.2"}

