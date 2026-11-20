import os
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI

BASE = Path(__file__).parent
STATIC = BASE / "static"
app = FastAPI(title="JARVIS iPhone")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

def get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="Serverda OPENAI_API_KEY sozlanmagan.")
    return OpenAI(api_key=key)

class ChatRequest(BaseModel):
    messages: list[dict]
    language: str = "auto"

@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")

@app.get("/manifest.webmanifest")
def manifest():
    return FileResponse(STATIC / "manifest.webmanifest", media_type="application/manifest+json")

@app.get("/sw.js")
def service_worker():
    return FileResponse(STATIC / "sw.js", media_type="application/javascript")

@app.post("/api/chat")
def chat(req: ChatRequest):
    client = get_client()
    system = (
        "You are JARVIS, a capable personal AI assistant. "
        "Understand Uzbek, Russian, and English. Reply in the same language as the user. "
        "Be warm, concise, and practical. Never claim to perform actions on the phone "
        "or other devices unless a connected tool actually performed them. "
        "Ask for confirmation before consequential actions."
    )
    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            instructions=system,
            input=req.messages[-30:]
        )
        return {"reply": response.output_text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"AI so'rovida xatolik: {str(e)}")

@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    client = get_client()
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="Audio yozuvi bo'sh.")
    suffix = ".m4a"
    content_type = audio.content_type or "audio/mp4"
    try:
        # OpenAI accepts common audio formats including mp3, mp4, mpeg, mpga, m4a, wav, and webm.
        import io
        stream = io.BytesIO(data)
        stream.name = audio.filename or "recording.m4a"
        result = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=stream
        )
        return {"text": result.text}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ovozni matnga aylantirishda xatolik: {str(e)}")
