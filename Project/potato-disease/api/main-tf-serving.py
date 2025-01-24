from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import os

app = FastAPI()
os.chdir(os.path.dirname(__file__))

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

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

    # !----- old code -----------
    # predictions = MODEL.predict(img_batch)
    # predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    # confidence = np.max(predictions[0])

    # return {"class": predicted_class, "confidence": float(confidence)}
    # !--------eixt old code--------

    response = requests.post(endpoint, json={"instances": img_batch.tolist()})
    prediction = response.json()["predictions"][0]
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])

    return {"class": predicted_class, "confidence": float(confidence)}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
