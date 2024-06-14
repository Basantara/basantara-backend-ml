import os
import uvicorn
import traceback
import tensorflow as tf
import numpy as np
import requests
from pydantic import BaseModel
from fastapi import FastAPI, Response, UploadFile
from utils import load_image_into_numpy_array

public_model_url = "https://storage.googleapis.com/tes_model/Model_1.zip"
local_model_path = "./"

def download_and_unzip_model(url, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    zip_path = os.path.join(dest_path, "model.zip")
    
    response = requests.get(url, stream=True)
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)
    
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_path)
    
    os.remove(zip_path)
    print(f"Model downloaded and extracted to {dest_path}")

download_and_unzip_model(public_model_url, local_model_path)

model = tf.saved_model.load("./saved_model")
model_signature = model.signatures["serving_default"]

class_name = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
}

app = FastAPI()

@app.get("/")
def index():
    return "Hello world!"

@app.post("/predict")
def predict_image(imageFile: UploadFile, response: Response):
    try:
        if imageFile.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return "File is Not an Image"
        
        image = load_image_into_numpy_array(imageFile.file.read())
        print("Image shape:", image.shape)
        

        img_height, img_width = 224, 224 
        image = tf.image.resize(image, (img_height, img_width))
        image = np.expand_dims(image, axis=0)
        image = image / 255.0 
        
        input_tensor = tf.convert_to_tensor(image)
        result = model_signature(input_tensor)
        
        predictions = result['dense_1'] 
        print(predictions)
        print(max(predictions[0]))
        
        predicted_class = np.argmax(predictions, axis=1)
        predicted_class_name = class_name.get(predicted_class[0])
        
        return {"predicted_class": int(predicted_class[0]), "class_name": predicted_class_name}
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"


port = os.environ.get("PORT", 8080)
print(f"Listening to http://0.0.0.0:{port}")
uvicorn.run(app, host='0.0.0.0', port=port)
