from fastapi import APIRouter, UploadFile, File
from PIL import Image
import io
from typing import List

router = APIRouter()

@router.post("/classify")
async def classify_dog(files: List[UploadFile] = File(...)):
    images = []

    try:
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            images.append(image)
    except Exception:
        return {"error": "Invalid image file"}