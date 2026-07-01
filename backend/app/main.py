from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app.generation import answer_question
from app.recommender import recommend

app = FastAPI(title="LithoMind API", version="1.0.0")

# Allows the React frontend (running on a different port during development)
# to actually call this API — browsers block cross-origin requests by default
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your actual frontend URL before deploying
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_endpoint(request: QuestionRequest):
    result = answer_question(request.question)
    return result

@app.post("/recommend")
async def recommend_endpoint(
    image: UploadFile,
    description: Optional[str] = Form(None)
):
    image_bytes = await image.read()
    result = recommend(image_bytes, description)
    return result

@app.get("/health")
async def health_check():
    return {"status": "ok"}