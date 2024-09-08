import os
import urllib.request
import zipfile

# Define the URL and the output path
tiny_imagenet_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
output_dir = "./data"
zip_file_path = os.path.join(output_dir, "tiny-imagenet-200.zip")
dataset_dir = os.path.join(output_dir, "tiny-imagenet-200")

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Download the Tiny ImageNet dataset
print("Downloading Tiny ImageNet dataset...")
urllib.request.urlretrieve(tiny_imagenet_url, zip_file_path)
print("Download complete.")

# Extract the dataset
print("Extracting the dataset...")
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(output_dir)
print("Extraction complete.")

# Check if the dataset is extracted successfully
if os.path.isdir(dataset_dir):
    print(f"Tiny ImageNet dataset is ready at {dataset_dir}")
else:
    print("Failed to extract the dataset correctly.")
