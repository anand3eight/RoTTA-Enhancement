import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from modules.helper import evaluate_tta
from modules.defense import RoTTA
import os

def build_optimizer(method = 'SGD'):
    def optimizer(params):
        if method == 'Adam':
            return torch.optim.Adam(params, lr=1e-3)
        
        elif method == 'SGD':
            return torch.optim.SGD(params, lr=1e-2)

        else:
            raise NotImplementedError

    return optimizer

def testTimeAdaptation(student, dataset_path, model_name, attack_type):
    
    batch_size = 32
    # model, optimizer

    optimizer = build_optimizer()

    tta_model = RoTTA(student, optimizer)
    tta_model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    clean_dataset_path = ''
    clean_dataset = datasets.ImageFolder(clean_dataset_path, transform=transform)
    clean_data_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False)
    tta_model.obtain_origin_stat(clean_data_loader)

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    evaluate_tta(loader, tta_model, student, model_name, attack_type)

def main():
    models_path = ''
    dataset = f""
    models = {'RN18-IBD' : 'trained_resnet_updated.pth'}  
    attacks = [d.name for d in os.scandir(dataset) if d.is_dir()]
    for attack in attacks :
        student = torch.load(f"{models_path}/{models['RN18-IBD']}")
        dataset_dir = f"{dataset}/{attack}"
        print(dataset_dir)
        testTimeAdaptation(student, dataset_dir, 'RN18-IBD', attack)

if __name__ == "__main__":
    main()
