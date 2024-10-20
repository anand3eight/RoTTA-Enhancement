import os
import tarfile
from torchvision.datasets.utils import download_url
import numpy as np
from PIL import Image
import pickle

# Define the download URL for CIFAR-100 dataset
url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
dataset_folder = "cifar-100"

# Function to download and extract dataset
def download_and_extract(url, dataset_folder):
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    tar_path = os.path.join(dataset_folder, "cifar-100-python.tar.gz")
    if not os.path.exists(tar_path):
        download_url(url, dataset_folder)
    
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=dataset_folder)

# Load batch file and get data
def load_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Save images to class folders
def save_images(data, labels, classes, folder_name):
    for i, (img_data, label) in enumerate(zip(data, labels)):
        class_name = classes[label]
        class_folder = os.path.join(folder_name, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        
        img_data = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img_data)
        img.save(os.path.join(class_folder, f"{i}.png"))

# Download and extract CIFAR-100
download_and_extract(url, dataset_folder)

# Load meta file for class names
meta_file = os.path.join(dataset_folder, "cifar-100-python", "meta")
meta_data = load_batch(meta_file)
fine_label_names = [x.decode('utf-8') for x in meta_data[b'fine_label_names']]

# Load train and test batches
train_file = os.path.join(dataset_folder, "cifar-100-python", "train")
train_data = load_batch(train_file)
train_images = train_data[b'data']
train_labels = train_data[b'fine_labels']

test_file = os.path.join(dataset_folder, "cifar-100-python", "test")
test_data = load_batch(test_file)
test_images = test_data[b'data']
test_labels = test_data[b'fine_labels']

# Save training and testing images into corresponding class folders
save_images(train_images, train_labels, fine_label_names, os.path.join(dataset_folder, "train"))
save_images(test_images, test_labels, fine_label_names, os.path.join(dataset_folder, "test"))

print("Images saved in class folders.")
