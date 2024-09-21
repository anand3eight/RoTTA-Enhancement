import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchattacks import PGD
from PIL import Image
import timm
from tqdm import tqdm

def generate_pgd_samples(folder_path, model, dataset_name, model_name):
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Define attack
    pgd_attack = PGD(model, eps=0.3, alpha=2/255, steps=10)
    
    # Create output folder
    output_folder = os.path.join('./PGD', dataset_name, model_name)
    os.makedirs(output_folder, exist_ok=True)

    # Map class indices to class names
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Generate and save PGD attack samples
    for i, (images, labels) in enumerate(tqdm(dataloader, desc="Generating PGD samples")):
        images, labels = images.to(device), labels.to(device)
        adv_images = pgd_attack(images, labels)
        
        # Save the adversarial image
        adv_image = adv_images[0].cpu().detach().numpy()
        adv_image = (adv_image * 255).transpose(1, 2, 0).astype('uint8')
        adv_image_pil = Image.fromarray(adv_image)

        # Get the class name for the current sample
        class_name = idx_to_class[labels.item()]
        class_output_folder = os.path.join(output_folder, class_name)
        os.makedirs(class_output_folder, exist_ok=True)
        
        adv_image_pil.save(os.path.join(class_output_folder, f"adv_image_{i}.png"))

    print(f"Adversarial samples saved in: {output_folder}")

# Example usage
dataset_name = 'CIFAR-10'
model_name = 'ViT'
folder_path = f'../Dataset/tiny/{dataset_name}/test'
model = timm.create_model('vit_base_patch16_224', pretrained=True)
generate_pgd_samples(folder_path, model, dataset_name, model_name)
