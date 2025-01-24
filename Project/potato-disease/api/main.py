from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
os.chdir(os.path.dirname(__file__))

origins = [
    "http://localhost",
    "http://localhost:3000",  # Allow frontend on port 3000
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of origins allowed to make requests
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

MODEL = tf.keras.models.load_model("../saved_models/1")


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, I am alive!"


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    return np.array(image)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {"class": predicted_class, "confidence": float(confidence)}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
