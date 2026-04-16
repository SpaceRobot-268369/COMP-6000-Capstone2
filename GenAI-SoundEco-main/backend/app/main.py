from __future__ import annotations

from fastapi import FastAPI

from app.api.routes.processing import router as processing_router


app = FastAPI(
    title="GenAI-SoundEco Backend API",
    version="0.1.0",
    description="API for processing audio files into acoustic features and VGGish embeddings.",
)

app.include_router(processing_router)
