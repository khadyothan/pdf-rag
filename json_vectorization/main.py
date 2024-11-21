import json
import os
import shutil
import requests
import argparse
from clip_vectorization import vectorize_text, vectorize_image

def download_from_google_drive(google_drive_url, destination_path):
    """
    Downloads a file from a Google Drive shareable link.

    Args:
        google_drive_url (str): The shareable Google Drive link.
        destination_path (str): The path where the file will be saved.
    """
    import re
    file_id_match = re.search(r'd/([^/]+)/', google_drive_url)
    if not file_id_match:
        print(f"Invalid Google Drive link: {google_drive_url}")
        return False
    
    file_id = file_id_match.group(1)
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        with open(destination_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"File downloaded successfully: {destination_path}")
        return True
    except requests.RequestException as e:
        print(f"Error downloading file from Google Drive: {e}")
        return False

def process_json(json_file, output_json):
    print("-" * 50)
    download_folder = "downloaded_images"
    os.makedirs(download_folder, exist_ok=True)  # Create a temporary folder for downloaded images
    
    # Read JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    embeddings = {}  # To store key-value pairs of embeddings

    # Iterate through papers in the JSON
    for paper_key, paper_content in data.items():
        print(f"Processing {paper_key}...")

        # Extract and vectorize abstract
        abstract = paper_content.get('abstract', 'No abstract provided')
        print(f"Abstract: {abstract}")
        embeddings[f"{paper_key}_abstract"] = vectorize_text(abstract)

        # Extract and vectorize subsection values
        sections = paper_content.get('sections', {})
        for section, value in sections.items():
            print(f"{section}: {value}")
            embeddings[f"{paper_key}_{section}"] = vectorize_text(value)

        # Download images and vectorize them
        images = paper_content.get('images', {})
        for image_key, image_content in images.items():
            image_desc = image_content.get('image_desc', 'No description')
            image_url = image_content.get('image_location', None)
            if image_url:
                image_path = os.path.join(download_folder, f"{paper_key}_{image_key}.jpg")
                if "drive.google.com" in image_url:
                    # Handle Google Drive links
                    print(f"Downloading {image_desc} from {image_url}...")
                    success = download_from_google_drive(image_url, image_path)
                    if not success:
                        print(f"Failed to download {image_url}")
                        continue
                else:
                    # Handle direct links
                    print(f"Downloading {image_desc} from {image_url}...")
                    try:
                        response = requests.get(image_url, stream=True)
                        response.raise_for_status()
                        with open(image_path, 'wb') as img_file:
                            for chunk in response.iter_content(1024):
                                img_file.write(chunk)
                        print(f"Saved to {image_path}")
                    except requests.RequestException as e:
                        print(f"Failed to download {image_url}: {e}")
                        continue

                # Vectorize the image and add to embeddings
                embeddings[f"{paper_key}_{image_key}"] = vectorize_image(image_path)

        print("-" * 50)
    
    # Save the embeddings to a JSON file
    with open(output_json, 'w') as outfile:
        json.dump(embeddings, outfile, indent=4)
    print(f"Embeddings saved to {output_json}")

    # Clean up the downloaded_images folder
    shutil.rmtree(download_folder)
    print(f"Temporary folder '{download_folder}' deleted.")

def main():
    parser = argparse.ArgumentParser(description="Process a JSON file and download images.")
    parser.add_argument("json_file", help="Path to the input JSON file.")
    parser.add_argument("output_json", help="Path to the output JSON file.")
    args = parser.parse_args()
    
    process_json(args.json_file, args.output_json)

if __name__ == "__main__":
    main()