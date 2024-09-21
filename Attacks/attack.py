import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchattacks import PGD, AutoAttack, CW, FGSM
from PIL import Image
import timm
from tqdm import tqdm

def generate_attack_samples(folder_path, model, dataset_name, model_name):
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Prepare dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=folder_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Define attack
    pgd_attack = PGD(model, eps=0.3, alpha=2/255, steps=10)
    auto_attack = AutoAttack(model, norm='Linf', eps=0.3)
    cw_attack = CW(model, c=1, kappa=0)
    fgsm_attack = FGSM(model, eps=0.15)


    output_folder = os.path.join(dataset_name, model_name)
    os.makedirs(output_folder, exist_ok=True)

    # Map class indices to class names
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Generate and save PGD attack samples
    for i, (images, labels) in enumerate(tqdm(dataloader, desc="Generating PGD samples")):
        images, labels = images.to(device), labels.to(device)

        adv_images_fgsm = fgsm_attack(images, labels) 
        adv_images_pgd = pgd_attack(images, labels)
        adv_images_auto = auto_attack(images, labels)
        adv_images_cw = cw_attack(images, labels)
               
        # Save images for each attack in separate folders
        for attack_type, adv_images in zip(['PGD', 'AutoAttack', 'CW', 'FGSM'], 
                                           [adv_images_pgd, adv_images_auto, adv_images_cw, adv_images_fgsm]):
            
            class_name = idx_to_class[labels.item()]

            # Create separate folder for each attack
            attack_output_folder = os.path.join(output_folder, attack_type)
            class_output_folder = os.path.join(attack_output_folder, class_name)
            os.makedirs(class_output_folder, exist_ok=True)

            adv_image = adv_images[0].cpu().detach().numpy()
            adv_image = (adv_image * 255).transpose(1, 2, 0).astype('uint8')
            adv_image_pil = Image.fromarray(adv_image)

            adv_image_pil.save(os.path.join(class_output_folder, f"adv_image_{i}.png"))

    print(f"Adversarial samples saved in: {output_folder}")

# Example usage
dataset_name = 'CIFAR-10'
model_name = 'ResNet18'
folder_path = f'../Dataset/tiny/{dataset_name}/test'
model = torch.load('../Training/Models/trained_resnet.pth')
generate_attack_samples(folder_path, model, dataset_name, model_name)
