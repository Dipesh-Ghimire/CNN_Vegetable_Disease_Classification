from fastapi import FastAPI, UploadFile, File
import numpy as np
from io import BytesIO
import uvicorn
# from PIL import Image, ImageOps
import tensorflow as tf
# import imageio
import imageio.v2 as imageio

app = FastAPI()



MODEL = tf.keras.models.load_model("../saved_model/1")


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
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return{
        'class':predicted_class,
        'confidence':float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, port=8000, host='localhost')
