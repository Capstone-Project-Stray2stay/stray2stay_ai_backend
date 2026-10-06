import io
import json
from pathlib import Path
import re
from typing import List

import pillow_heif
import timm
import torch
from fastapi import APIRouter, File, UploadFile
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

pillow_heif.register_heif_opener()

router = APIRouter()
SPECIALIZED_CONFIDENCE_THRESHOLD = 0.40

BASE_DIR = Path(__file__).resolve().parent.parent
cat_model_path = BASE_DIR / "models" / "cat_model"
cat_breed_model_path = BASE_DIR / "models" / "cat_british_american_scottish"

cat_processor = AutoImageProcessor.from_pretrained(str(cat_model_path), use_fast=False)
cat_model = AutoModelForImageClassification.from_pretrained(str(cat_model_path))
cat_model.eval()

with (cat_breed_model_path / "config.json").open(encoding="utf-8") as config_file:
    cat_breed_config = json.load(config_file)

with (cat_breed_model_path / "preprocessor_config.json").open(encoding="utf-8") as processor_file:
    cat_breed_processor_config = json.load(processor_file)

cat_breed_model = timm.create_model(
    cat_breed_config["architecture"],
    pretrained=False,
    num_classes=cat_breed_config["num_classes"],
    drop_path_rate=cat_breed_config.get("drop_path_rate", 0.0),
    img_size=cat_breed_config.get("img_size", 224),
)
weights_file = cat_breed_config.get("weights_file", "model.safetensors")
weights_path = cat_breed_model_path / weights_file
if not weights_path.exists():
    weights_path = cat_breed_model_path / "model.safetensors"
cat_breed_model.load_state_dict(
    load_file(str(weights_path), device="cpu")
)
cat_breed_model.eval()

cat_breed_id2label = {
    str(index): label for index, label in cat_breed_config["id2label"].items()
}
cat_breed_labels = [cat_breed_id2label[str(index)] for index in range(cat_breed_config["num_classes"])]
cat_breed_label_keys = {
    "".join(character.lower() for character in label if character.isalnum())
    for label in cat_breed_labels
}
cat_breed_display_labels = {
    "".join(character.lower() for character in label if character.isalnum()): re.sub(
        r"(?<=[a-z])(?=[A-Z])", " ", label
    )
    for label in cat_breed_labels
}

processor_interpolation = getattr(
    transforms.InterpolationMode,
    cat_breed_processor_config.get("interpolation", "bilinear").upper(),
    transforms.InterpolationMode.BILINEAR,
)
cat_breed_processor = transforms.Compose([
    transforms.Resize(cat_breed_processor_config["resize"], interpolation=processor_interpolation),
    transforms.CenterCrop(cat_breed_processor_config["center_crop"]),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=tuple(cat_breed_processor_config["mean"]),
        std=tuple(cat_breed_processor_config["std"]),
    ),
])


def normalize_breed_label(label: str) -> str:
    compact_label = "".join(character.lower() for character in label if character.isalnum())
    return cat_breed_display_labels.get(compact_label, label)


@router.post("/classify")
async def classify_cat(files: List[UploadFile] = File(...)):
    images = []

    try:
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            images.append(image)
    except Exception as e:
        return {"error": f"Invalid image file: {str(e)}"}

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
            "label": normalize_breed_label(cat_model.config.id2label[idx]),
            "confidence": round(float(topk.values[0][i]), 4)
        })

    use_specialized_model = any(
        "".join(character.lower() for character in prediction["label"] if character.isalnum())
        in cat_breed_label_keys
        and prediction["confidence"] >= SPECIALIZED_CONFIDENCE_THRESHOLD
        for prediction in results
    )
    if use_specialized_model:
        specialized_inputs = torch.stack([cat_breed_processor(image) for image in images])

        with torch.no_grad():
            specialized_logits = cat_breed_model(specialized_inputs)

        specialized_probs = torch.nn.functional.softmax(specialized_logits, dim=-1)
        specialized_avg_probs = specialized_probs.mean(dim=0, keepdim=True)
        specialized_topk = torch.topk(specialized_avg_probs, k=len(cat_breed_labels))

        results = []
        for i in range(len(cat_breed_labels)):
            idx = specialized_topk.indices[0][i].item()
            results.append({
                "label": normalize_breed_label(cat_breed_labels[idx]),
                "confidence": round(float(specialized_topk.values[0][i]), 4)
            })

    return {
        "num_images": len(images),
        "predictions": results
    }


@router.post("/classify/test")
async def classify_cat_british_american_scottish(files: List[UploadFile] = File(...)):
    images = []

    try:
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            images.append(image)
    except Exception as e:
        return {"error": f"Invalid image file: {str(e)}"}

    inputs = torch.stack([cat_breed_processor(image) for image in images])

    with torch.no_grad():
        logits = cat_breed_model(inputs)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    avg_probs = probs.mean(dim=0, keepdim=True)
    topk = torch.topk(avg_probs, k=len(cat_breed_labels))

    results = []
    for i in range(len(cat_breed_labels)):
        idx = topk.indices[0][i].item()
        results.append({
            "label": normalize_breed_label(cat_breed_labels[idx]),
            "confidence": round(float(topk.values[0][i]), 4)
        })

    return {
        "num_images": len(images),
        "predictions": results
    }
