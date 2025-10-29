import os
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from . import model as skin_model

API_KEY = os.getenv("API_KEY")  # optional

app = FastAPI(
    title="Skin Condition Predictor (Research-Only)",
    version="0.1.0",
    description="Uploads patient metadata + lesion image and returns top predicted skin conditions. Not a medical device."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _load():
    try:
        skin_model.initialize()
    except Exception as e:
        print(f"Startup warning: {e}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    blood_pressure: str = Form(...),
    symptoms: str = Form(""),
    medications: str = Form(""),
    api_key: str = Form(default="")
):
    if API_KEY and api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if age < 0 or age > 120:
        raise HTTPException(status_code=400, detail="Invalid age")
    if image.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Unsupported image type")

    try:
        img_bytes = await image.read()
        results = skin_model.predict(img_bytes, top_k=3)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    top = results[0] if results else None

    return JSONResponse({
        "prediction": top,
        "top_k": results,
        "disclaimer": (
            "This tool is for research and educational purposes only, not for diagnosis or treatment. "
            "Always consult a qualified clinician."
        ),
        "echo": {
            "name": name,
            "age": age,
            "blood_pressure": blood_pressure,
            "symptoms": symptoms,
            "medications": medications
        }
    })
