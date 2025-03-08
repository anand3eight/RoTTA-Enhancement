import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchattacks import PGD, CW, APGD, APGDT, FAB, Square, JSMA
from PIL import Image
from tqdm import tqdm
from adaptive_attacks import mixed_batch_attack

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

    # Ignore ipynb_checkpoints folder
    dataset = datasets.ImageFolder(root=folder_path, transform=transform)
    dataset.samples = [s for s in dataset.samples if "ipynb_checkpoints" not in s[0]]

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Define attacks
    pgd_attack = PGD(model, eps=0.3, alpha=2/255, steps=10)
    cw_attack = CW(model, c=1, kappa=0)
    apgd_attack = APGD(model, eps=8/255, steps=10)
    apgdt_attack = APGDT(model, eps=8/255, steps=10)
    fab_attack = FAB(model, eps=8/255, steps=10)
    square_attack = Square(model, eps=8/255)
    jsma_attack = JSMA(model, theta=1.0, gamma=0.1)

    # Output folders
    output_folder = os.path.join(dataset_name, model_name)
    os.makedirs(output_folder, exist_ok=True)

    clean_output_folder = os.path.join(output_folder, "Clean")
    os.makedirs(clean_output_folder, exist_ok=True)

    # Map class indices to class names
    class_to_idx = dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Generate clean & attack samples
    for i, (images, labels) in enumerate(tqdm(dataloader, desc="Processing images")):
        images, labels = images.to(device), labels.to(device)

        # Save clean images
        class_name = idx_to_class[labels.item()]
        class_clean_folder = os.path.join(clean_output_folder, class_name)
        os.makedirs(class_clean_folder, exist_ok=True)

        clean_image = images[0].cpu().detach().numpy()
        clean_image = (clean_image * 255).transpose(1, 2, 0).astype('uint8')
        clean_image_pil = Image.fromarray(clean_image)

        clean_image_pil.save(os.path.join(class_clean_folder, f"clean_image_{i}.png"))

        # Generate attack samples
        adv_images_list = {
        #     "PGD": pgd_attack(images, labels),
        #     "CW": cw_attack(images, labels),
        #     "APGD": apgd_attack(images, labels),
        #     "APGD_T": apgdt_attack(images, labels),
        #     "FAB": fab_attack(images, labels),
        #     "Square": square_attack(images, labels),
              "MBA" : mixed_batch_attack(model, images, labels),
         }

        # Save attack images
        for attack_type, adv_images in adv_images_list.items():
            attack_output_folder = os.path.join(output_folder, attack_type, class_name)
            os.makedirs(attack_output_folder, exist_ok=True)

            adv_image = adv_images[0].cpu().detach().numpy()
            adv_image = (adv_image * 255).transpose(1, 2, 0).astype('uint8')
            adv_image_pil = Image.fromarray(adv_image)

            adv_image_pil.save(os.path.join(attack_output_folder, f"adv_image_{i}.png"))

    print(f"Clean & adversarial samples saved in: {output_folder}")

# Example usage
dataset_name = 'CIFAR-10'
model_name = 'ResNet18'
folder_path = ''
model_path = ''
model = torch.load( model_path )
generate_attack_samples(folder_path, model, dataset_name, model_name)
