import os
from torchvision.datasets import CIFAR10

# Define the path where you want to save the images
output_dir = 'CIFAR-10'
train_dir = os.path.join(output_dir, 'train')
test_dir = os.path.join(output_dir, 'test')

# CIFAR-10 class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck']

# Create directory structure for train and test sets
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

# Load CIFAR-10 training data
train_dataset = CIFAR10(root='./data', train=True, download=True)
for idx, (image, label) in enumerate(train_dataset):
    # Directly use the PIL image
    # Define the image path
    img_path = os.path.join(train_dir, classes[label], f"{idx}.png")
    # Save the image
    image.save(img_path)

# Load CIFAR-10 test data
test_dataset = CIFAR10(root='./data', train=False, download=True)
for idx, (image, label) in enumerate(test_dataset):
    # Directly use the PIL image
    # Define the image path
    img_path = os.path.join(test_dir, classes[label], f"{idx}.png")
    # Save the image
    image.save(img_path)

print("CIFAR-10 dataset saved in image form.")

