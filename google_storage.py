import os
from google.cloud import storage

def upload_image(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    if blob.exists():
        base, extension = os.path.splitext(destination_blob_name)
        counter = 1
        while blob.exists():
            new_blob_name = f"{base}_{counter}{extension}"
            blob = bucket.blob(new_blob_name)
            counter += 1
        destination_blob_name = new_blob_name

    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")
