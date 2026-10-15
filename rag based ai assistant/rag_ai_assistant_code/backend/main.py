import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import auth, student, teacher
from backend.services import db

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await db.connect_db()
        app.state.db_available = True
    except Exception as e:
        import logging

        logging.warning("Database connection failed; continuing without DB: %s", e)
        app.state.db_available = False
    yield
    if getattr(app.state, "db_available", False):
        await db.close_db()


app = FastAPI(
    title="AI Teaching Assistant API",
    description="Modular RAG backend for teachers and students",
    version="1.0.0",
    lifespan=lifespan,
)

origins = os.environ.get("CORS_ORIGINS", "http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(teacher.router)
app.include_router(student.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
