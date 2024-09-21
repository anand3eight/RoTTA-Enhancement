import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from helper import evaluate_tta


def get_loader(dir) :
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    return dataloader

if __name__ == "__main__" :
    model = torch.load('../Training/Models/trained_resnet.pth')
    model.cuda()
    model_name = 'ResNet18'
    attacks = ['FGSM', 'PGD', 'AutoAttack', 'CW']
    for attack_type in attacks :
        test_dir = f'../Attacks/CIFAR-10/ResNet18/{attack_type}'
        loader = get_loader(test_dir)
        evaluate_tta(loader, model, model_name, attack_type)
