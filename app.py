import torch
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import JSONResponse
import numpy as np
import torch.nn as nn
import os
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import psutil
import time
from typing import Optional
from tqdm import tqdm
import logging

app = FastAPI(title="EfficientNetV2 Image Classification API")

# Define class names
CLASS_NAMES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_v2_s(pretrained=False)
# Modify the classifier to match the training architecture
num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(num_ftrs, len(CLASS_NAMES))
)
model = model.to(device)
model.load_state_dict(torch.load("efficientnet_v2_model.pth", map_location=device))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

@app.get("/")
async def root():
    return {"message": "EfficientNetV2 Image Classification API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and transform image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return JSONResponse({
            "predicted_class": int(predicted_class),
            "predicted_emotion": CLASS_NAMES[predicted_class],
            "confidence": float(confidence)
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/evaluate")
async def evaluate(
    validation_dir: Optional[str] = Body(default="/app/validation", embed=True)
):
    """
    Evaluate the model on a validation dataset. The directory should have subfolders for each class.
    """
    class_names = CLASS_NAMES
    y_true, y_pred = [], []
    inference_times = []
    image_info = []
    all_image_paths = []
    all_labels = []

    # Gather all image paths and their true labels
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(validation_dir, class_name)
        image_paths = glob.glob(os.path.join(class_dir, "*.jpg"))
        logging.info(f"Found {len(image_paths)} images in {class_name}")
        all_image_paths.extend(image_paths)
        all_labels.extend([idx] * len(image_paths))

    total_images = len(all_image_paths)
    if total_images == 0:
        return JSONResponse(
            status_code=400,
            content={"error": "No images found in the validation directory. Please check your path and data."}
        )

    logging.info(f"Starting evaluation on {total_images} images...")

    for img_path, true_label in tqdm(zip(all_image_paths, all_labels), total=total_images, desc="Evaluating", ncols=80):
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            start_time = time.time()
            with torch.no_grad():
                outputs = model(image_tensor)
                pred = torch.argmax(outputs, dim=1).item()
            end_time = time.time()
            inference_times.append(end_time - start_time)
            y_true.append(true_label)
            y_pred.append(pred)
            image_info.append({"file": img_path, "true": true_label, "pred": pred})
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
            continue

    # Classification metrics
    labels = list(range(len(class_names)))
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0, labels=labels)
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    class_report = classification_report(
        y_true, y_pred, target_names=class_names, labels=labels, output_dict=True, zero_division=0
    )

    # Efficiency metrics
    avg_inference_time = float(np.mean(inference_times)) if inference_times else None
    throughput = float(total_images / sum(inference_times)) if inference_times else None
    process = psutil.Process(os.getpid())
    memory_usage_mb = process.memory_info().rss / 1024 / 1024
    model_size_mb = os.path.getsize("efficientnet_v2_model.pth") / 1024 / 1024

    def sanitize_for_json(obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        else:
            return obj

    result = {
        "performance": {
            "accuracy": accuracy,
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "f1_per_class": f1.tolist(),
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report
        },
        "efficiency": {
            "avg_inference_time_sec": avg_inference_time,
            "throughput_img_per_sec": throughput,
            "memory_usage_mb": memory_usage_mb,
            "model_size_mb": model_size_mb
        },
        "n_images": total_images,
        "classes": class_names
    }
    logging.info("Evaluation complete.")
    return sanitize_for_json(result)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080) 