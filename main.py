from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Form
from pydantic import BaseModel
from typing import List
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf

app = FastAPI()

origins = ["*"]  # Set this to your specific frontend domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "./saved_models/new"  # Relative path to the saved_models folder

# Load the model
MODEL = tf.keras.models.load_model(model_path)
CLASS_NAMES = ['Cadang_Cadang', 'Caterpillars', 'Coconut_Bud_Rot', 'Coconut_Leaf_Spot', 'Coconut_Scale_Insect', 'Drying_of_Leaflets', 'Normal_Coconut', 'Yellowing']

class PredictionResult(BaseModel):
    class1: str
    confidence1: float
    # class2: str
    # confidence2: float

@app.get("/ping")
def ping():
    return "Test Ping....."

@app.get("/")
def root():
    return {"message": "Hello World!!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = image.resize((256, 256))  # Resize the image to match the expected input shape

    img_array = np.array(image)  # Convert the PIL image to a NumPy array
    img_batch = np.expand_dims(img_array, 0)  # Add an extra dimension for the batch

    predictions = MODEL.predict(img_batch)

    top_classes = [CLASS_NAMES[idx] for idx in np.argsort(predictions[0])[-1:][::-1]]
    top_confidences = [float(conf) * 100 for conf in np.sort(predictions[0])[-1:][::-1]]

    result = PredictionResult(class1=top_classes[0], confidence1=top_confidences[0])
    return result

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
