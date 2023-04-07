from fastapi import FastAPI, UploadFile, File
import numpy as np
import requests
from io import BytesIO
import uvicorn
# from PIL import Image, ImageOps
import tensorflow as tf
# import imageio
import imageio.v2 as imageio

app = FastAPI()

# dynamically loads latest version
endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"


CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]


# Takes data:byte as argument and returns np array
def read_file_as_image(data: bytes) -> np.ndarray:
    img = imageio.imread(BytesIO(data), pilmode='RGB')
    return img


@app.get("/ping")
async def ping():
    x = 9
    return "Hello DJ, I am alive"


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    # image:Array:[256,256,3]
    # predict takes [[256,256,3]] so another dimension is to be added
    img_batch = np.expand_dims(image,0)
    json_data = {
        "instances": img_batch.tolist()
    }
    response = requests.post(endpoint,json=json_data)
    prediction = np.array(response.json()["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return{
        "class":predicted_class,
        "confidence":confidence
    }


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host='localhost')
