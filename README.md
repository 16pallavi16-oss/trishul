# Trishul Document Q&A

A local document question-answering app. The FastAPI backend handles uploads, ingestion, search, and chat; the Next.js frontend provides the chat and upload pages.

## Requirements

- Python 3.12
- Node.js and npm
- Docker Desktop with Docker Compose
- Ollama, with the chat and embedding models available locally
- Tesseract OCR installed and available on `PATH` for image OCR

## Configure the backend

Run commands from the repository root (`D:\Trishul\trishul` on the original Windows setup).

Create and activate a virtual environment, then install the Python packages:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a root `.env` file. Keep secrets local; `.env` is ignored by Git. At minimum, set the database connection to the Postgres port exposed by `docker-compose.yml` and provide a Meilisearch key:

```dotenv
DATABASE_URL=postgresql://myuser:mypassword@localhost:5433/mydb
MEILI_MASTER_KEY=replace-with-a-local-development-key
```

The backend also supports `QDRANT_URL`, `MEILI_URL`, `REDIS_URL`, `OLLAMA_URL`, `CHAT_MODEL`, `EMBED_MODEL`, and `OLLAMA_NUM_PREDICT`. Set `OLLAMA_NUM_PREDICT` to a positive integer to cap generated answer tokens (for example, `OLLAMA_NUM_PREDICT=256`); leave it unset to use Ollama's default. Defaults are local service URLs and the model names in `platforms/chat.py` and `platforms/embed.py`.

Start the local data services:

```powershell
docker compose up -d
```

Make sure Ollama is running and has the configured chat and embedding models. Then start the API from the repository root:

```powershell
python -m uvicorn main:app --reload
```

The API is at <http://localhost:8000>; its interactive docs are at <http://localhost:8000/docs>.

## Run the frontend

In a second terminal:

```powershell
cd next-js
npm install
npm run dev
```

Open <http://localhost:3000/chat> to chat, or <http://localhost:3000/upload> to upload documents. Keep the API and frontend terminals running while using the app.

If accessing the frontend from another device on your network, start the backend with `python -m uvicorn main:app --reload --host 0.0.0.0` so that device can reach the API.

## Common startup issue

If FastAPI reports that form data requires `python-multipart`, install the project requirements in the active virtual environment:

```powershell
python -m pip install -r requirements.txt
```
