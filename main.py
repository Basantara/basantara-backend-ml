import os
import uvicorn
import traceback
import tensorflow as tf
import numpy as np
import requests
import string
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, Response, UploadFile
from utils import load_image_into_numpy_array
from google_storage import upload_image
from dotenv import load_dotenv
load_dotenv()

model_url = os.environ.get('MODEL_URL')
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
    
def download_modelh5(url, dest_path):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    model_path = os.path.join(dest_path, "model.h5")
    
    response = requests.get(url, stream=True)
    with open(model_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)
    
    print(f"Model downloaded to {dest_path}")
    
# Threshold Logic
def is_valid_prediction(prediction, threshold=0.5):
    confidence = np.max(prediction)
    return confidence > threshold

# Uncomment if the model is in cloud
# download_and_unzip_model(model_url, local_model_path) # saved_model format
download_modelh5(model_url, local_model_path) # h5 format

# if use saved model
# model = tf.saved_model.load("./saved_model")
# model_signature = model.signatures["serving_default"]

# If use h5 type uncomment line below
model = tf.keras.models.load_model('./model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

class_name = list(string.ascii_uppercase)

app = FastAPI()

@app.get("/")
def index():
    return "Hello world!"

@app.post("/predict")
def predict_image(imageFile: UploadFile, response: Response):
    try:
        if imageFile.content_type not in ["image/jpeg", "image/png"]:
            response.status_code = 400
            return "File is not an image"
        
        image = load_image_into_numpy_array(imageFile.file.read())
        
        img_height, img_width = 224, 224 
        image = image.resize((img_height, img_width))
        image = np.array(image)
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.vgg16.preprocess_input(image)
        
        predictions = model.predict(image)
        print(predictions)
        
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_name[predicted_class]
        confidence_score = np.max(predictions, axis=1)[0]
        
        # Store Image to Cloud Storage uncomment if in prod
        # local_image_path = f"./uploads/{predicted_class_name}/{imageFile.filename}"
        # os.makedirs(os.path.dirname(local_image_path), exist_ok=True)
        # with open(local_image_path, "wb") as f:
        #     f.write(imageFile.file.read())
            
        # bucket_name = os.environ.get('BUCKET_NAME')
        # destination_blob_name = f"uploads/{predicted_class_name}/{imageFile.filename}"
        # upload_image(bucket_name, local_image_path, destination_blob_name)
        # os.remove(local_image_path)
        if is_valid_prediction(confidence_score):
            return {
                "predicted_class": int(predicted_class), 
                "class_name": predicted_class_name,
                "confidence_score": float(confidence_score)
            }
        else:
            return {
                "message": "Confidence Score Under Threshold",
                "confidence_score": float(confidence_score)
            }
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"Listening to http://0.0.0.0:{port}")
    uvicorn.run(app, host='0.0.0.0', port=port)
