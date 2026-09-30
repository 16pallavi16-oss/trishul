from fastapi import FastAPI, HTTPException
import requests 
import shutil, tempfile
from fastapi.middleware.cors import CORSMiddleware
from platforms.db import init_db
from platforms.search import hybrid_search
from pydantic import BaseModel
from platforms.chat import answer_question
from platforms.threads import init_thread_db
from fastapi.responses import StreamingResponse
from platforms.chat import answer_question_stream
from pathlib import Path
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse
from platforms.jobs import start_ingest_job, get_job, stream_job_progress
from platforms.threads import clear_thread, save_feedback, get_feedback_summary



app = FastAPI()

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"

PROMPT = "say something about urself"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    thread_id: str
    k: int = 5
    method: str = "rrf"

@app.on_event("startup")
def on_startup():
    init_db()
    init_thread_db()

@app.post("/chat")
def chat(req: ChatRequest):
    return answer_question(req.question, req.thread_id, k=req.k, method=req.method)

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

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    return StreamingResponse(
        answer_question_stream(req.question, req.thread_id, k=req.k, method=req.method),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/ingest/upload")
async def ingest_upload(files: list[UploadFile] = File(...)):
    tmp = Path(tempfile.mkdtemp(prefix="ingest_"))
    for f in files:
        dest = tmp / f.filename          # filename carries the folder path
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)
    return {"job_id": start_ingest_job(tmp)}

@app.get("/ingest/jobs/{job_id}")
def ingest_job(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job

@app.get("/ingest/jobs/{job_id}/stream")
def ingest_job_stream(job_id: str):
    return StreamingResponse(
        stream_job_progress(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

class FeedbackRequest(BaseModel):
    message_id: int
    rating: str          # "up" | "down"
    comment: str | None = None

@app.delete("/chat/{thread_id}")
def clear(thread_id: str):
    return {"cleared": clear_thread(thread_id)}

@app.post("/feedback", status_code=201)
def feedback(req: FeedbackRequest):
    try:
        return {"id": save_feedback(req.message_id, req.rating, req.comment)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/feedback/summary")
def feedback_summary():
    return get_feedback_summary()
