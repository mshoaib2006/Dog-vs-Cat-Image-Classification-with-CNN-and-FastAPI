from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow import keras
import numpy as np
import cv2
import uvicorn


model = keras.models.load_model("dogcat_model.keras")

app = FastAPI()


def predict_image(image_bytes):
    
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


    img = cv2.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)  

   
    img = img.astype("float32") / 255.0

    prediction = model.predict(img)
    label = "Dog " if prediction[0][0] > 0.5 else "Cat "

    return label, float(prediction[0][0])

@app.post("/predict")
async def classify_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    label, confidence = predict_image(image_bytes)

    return JSONResponse({
        "filename": file.filename,
        "prediction": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
