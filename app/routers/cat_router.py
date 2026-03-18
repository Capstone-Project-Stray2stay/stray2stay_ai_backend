from fastapi import APIRouter, UploadFile, File
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
import io
from typing import List

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cat_model_path = os.path.join(BASE_DIR, "models", "cat_model")

cat_processor = AutoImageProcessor.from_pretrained(cat_model_path)
cat_model = AutoModelForImageClassification.from_pretrained(cat_model_path)
cat_model.eval()


@router.post("/classify")
async def classify_cat(files: List[UploadFile] = File(...)):
    images = []

    try:
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            images.append(image)
    except Exception:
        return {"error": "Invalid image file"}

    inputs = cat_processor(images=images, return_tensors="pt")

    with torch.no_grad():
        outputs = cat_model(**inputs)
        logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=-1)

    avg_probs = probs.mean(dim=0, keepdim=True)

    k = 3
    topk = torch.topk(avg_probs, k)

    results = []
    for i in range(k):
        idx = topk.indices[0][i].item()
        results.append({
            "label": cat_model.config.id2label[idx],
            "confidence": round(float(topk.values[0][i]), 4)
        })

    return {
        "num_images": len(images),
        "predictions": results
    }