# AI Resume Backend (FastAPI)

## Local

- Install: `pip install -r requirements.txt`
- Run: `uvicorn main:app --reload --host 127.0.0.1 --port 8000`

## Deploy (Render)

This repo includes `render.yaml` for a Render Blueprint deployment.

- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Build command: `pip install -r requirements.txt`

Environment variables:

- Copy `.env.example` → `.env` locally.
- In Render dashboard, set the same variables in the service environment.
