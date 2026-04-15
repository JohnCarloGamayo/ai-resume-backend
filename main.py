from pathlib import Path
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.evaluate import router as evaluate_router

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = FastAPI(
    title="Resume Evaluation and ATS Optimization API",
    version="1.0.0",
)

# Configure allowed origins from environment for deployment flexibility.
# Use comma-separated values in CORS_ALLOW_ORIGINS, or "*" to allow all.
raw_allowed_origins = os.getenv("CORS_ALLOW_ORIGINS", "*")
if raw_allowed_origins.strip() == "*":
    allowed_origins = ["*"]
else:
    allowed_origins = [
        origin.strip()
        for origin in raw_allowed_origins.split(",")
        if origin.strip()
    ]
    if not allowed_origins:
        allowed_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "https://ai-resume-frontend-johncarlogamayos-projects.vercel.app",
            "https://ai-resume-frontend-bjidqbdju-johncarlogamayos-projects.vercel.app",
        ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(evaluate_router)
