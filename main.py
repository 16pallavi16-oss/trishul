from fastapi import FastAPI
import requests 
from fastapi.middleware.cors import CORSMiddleware
from platforms.search import hybrid_search

app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"

PROMPT = "say something about urself"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/hello")
def hello():
    data = {
        "model": MODEL_NAME,
        "prompt": PROMPT,
        "stream": False 
    }
    response = requests.post(OLLAMA_URL, json=data)
    result = response.json()
    return {"ollama": result["response"]}

@app.get("/search")
def search(q: str, k: int = 5, method: str = "rrf", pool: int = 20, dense_weight: float = 0.5, lexical_weight: float = 0.5):
    return hybrid_search(q, k=k, method=method, pool=pool, dense_weight=dense_weight, lexical_weight=lexical_weight)

