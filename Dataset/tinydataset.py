import os
import shutil
from tqdm import tqdm

def copy_n_images(source_folder, destination_folder, class_name, n=200):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # List all files in the source folder
    source_folder = os.path.join(source_folder, class_name)
    files = os.listdir(source_folder)
    
    # Filter out only image files (you can add more extensions if needed)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    # Copy the first 200 images with progress bar
    for file_name in tqdm(image_files[:n], desc="Copying images", unit="file"):
        src_file_path = os.path.join(source_folder, file_name)
        dest_file_path = os.path.join(destination_folder, file_name)
        shutil.copy(src_file_path, dest_file_path)
    
    print(f"Copied {min(n, len(image_files))} images to {destination_folder}.")

# Example usage
n = 20
source_folder = 'cifar-100/test'  # Replace with the path to your source folder
folder = 'tiny/cifar-100/test'  # Replace with the path to your destination folder
for class_name in os.listdir(source_folder) :
    destination_folder = os.path.join(folder, class_name)
    copy_n_images(source_folder, destination_folder, class_name, n)
