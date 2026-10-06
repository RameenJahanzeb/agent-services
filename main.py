import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import build_graph

load_dotenv(Path(__file__).with_name(".env"))


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str
    operation: str
    success: bool


origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

api = FastAPI(title="Contact Manager Agent")

graph = build_graph()


@api.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="Message must not be empty.")

    try:
        result = await graph.ainvoke({"message": request.message.strip()})
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    return ChatResponse(
        reply=result["reply"],
        operation=result["operation"],
        success=result["success"],
    )


app = CORSMiddleware(
    app=api,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
