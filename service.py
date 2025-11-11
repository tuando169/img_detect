from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from typing import Dict
import io, os, torch
from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_NAME = os.getenv("MODEL_NAME", "strangerguardhf/nsfw_image_detection")

app = FastAPI(title="NSFW Image Detection API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)

model = None
processor = None
labels = None

def load_model():
    global model, processor, labels
    if model is not None:
        return
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = SiglipForImageClassification.from_pretrained(MODEL_NAME)
    id2label = getattr(model.config, "id2label", None)
    labels = [] if labels is None else labels
    if id2label:
        labels = [id2label.get(str(i), id2label.get(i, f"class_{i}")) for i in sorted(map(int, id2label.keys()))]
    else:
        labels = [f"class_{i}" for i in range(model.config.num_labels)]

@app.on_event("startup")
def _startup():
    load_model()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"ok": True, "model": MODEL_NAME, "labels": labels}

def classify_bytes(image_bytes: bytes) -> Dict:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
    nsfw_keys = {"Pornography", "Hentai", "Explicit Nudity", "Sexy or Nude", "Enticing or Sensual"}
    is_nsfw = any(scores.get(k, 0.0) >= 0.70 for k in nsfw_keys)
    top_label = max(scores, key=scores.get)
    return {"scores": scores, "top_label": top_label, "is_nsfw": is_nsfw}

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    data = await image.read()
    try:
        result = classify_bytes(data)
        return JSONResponse({"ok": True, **result})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
