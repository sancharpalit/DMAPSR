import requests
import os
import zipfile

def download_file_from_google_drive(file_id, dest_path):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    # Get initial response
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, dest_path)
    print(f"Downloaded to {dest_path}")

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # Filter out keep-alive chunks
                f.write(chunk)

# ID from shared Google Drive link
file_id = "1zKj8k1M-0VG-DvXqzKdBMnY8ZMYA4C4T"
destination = "LOLdataset_deep_retinex.zip"

# Create output directory
os.makedirs("LOL_dataset", exist_ok=True)
destination_path = os.path.join("LOL_dataset", destination)

# Download
download_file_from_google_drive(file_id, destination_path)

# Extract
print("Extracting...")
with zipfile.ZipFile(destination_path, 'r') as zip_ref:
    zip_ref.extractall("LOL_dataset")

print("Done! Extracted to LOL_dataset/")
