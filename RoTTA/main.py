import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from helper import evaluate_tta

from rotta import RoTTA
from tqdm import tqdm

def build_optimizer(method = 'Adam'):
    def optimizer(params):
        if method == 'Adam':
            return torch.optim.Adam(params, lr=1e-3)
        
        elif method == 'SGD':
            return torch.optim.SGD(params, lr=1e-3)

        else:
            raise NotImplementedError

    return optimizer

def testTimeAdaptation(base_model, dataset_path, attack_type):
    print("-" * 30)
    print("In the TTA Function")

    batch_size = 32
    # model, optimizer
    model = base_model
    params = model.parameters()

    optimizer = build_optimizer()

    tta_model = RoTTA(model, optimizer)
    if tta_model.optimizer == None : 
        return None
    tta_model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    evaluate_tta(loader, tta_model, 'RoTTA', attack_type)

def main():
    print("-" * 30)
    print("Setting up the Base Model : Resnet-18 and PGD attacked CIFAR-10 Dataset")
    base_model = torch.load('../Training/Models/trained_resnet.pth')
    base_model.cuda()
    model_name = 'ResNet18'
    attack_type = 'AutoAttack'
    test_dir = f'../Attacks/CIFAR-10/{model_name}/{attack_type}'
    testTimeAdaptation(base_model, test_dir, attack_type)

if __name__ == "__main__":
    main()