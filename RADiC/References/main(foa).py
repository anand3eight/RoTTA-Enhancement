import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from helper import evaluate_tta
from rotta import RoTTA
from WRN import *

def build_optimizer(method = 'SGD'):
    def optimizer(params):
        if method == 'Adam':
            return torch.optim.Adam(params, lr=1e-1)
        
        elif method == 'SGD':
            return torch.optim.SGD(params, lr=1e-1)

        else:
            raise NotImplementedError

    return optimizer

def testTimeAdaptation(student, dataset_path, attack_type):
    
    batch_size = 32
    # model, optimizer

    optimizer = build_optimizer()

    tta_model = RoTTA(student, optimizer, attack_type)
    tta_model.cuda()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    evaluate_tta(loader, tta_model, 'RoTTA+FOA', attack_type)

def main():
    models_path = '../Training/Models'
    dataset = f"../Attacks"
    models = {'Resnet18' : 'trained_resnet.pth', 'MobileNetV2' : 'trained_mobilenetv2.pth'}
    attacks = ['FGSM', 'PGD', 'CW', 'AutoAttack']
    for model_path in models :
        for attack in attacks :
            student = torch.load(f'{models_path}/{models[model_path]}')
            dataset_dir = f"{dataset}/{model_path}/{attack}"
            print(dataset_dir)
            # testTimeAdaptation(student, dataset_dir, attack)

if __name__ == "__main__":
    main()