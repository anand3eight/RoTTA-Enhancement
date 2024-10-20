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
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    clean_dataset_path = '../Dataset/cifar-100/train'
    clean_dataset = datasets.ImageFolder(clean_dataset_path, transform=transform)
    clean_data_loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False)
    tta_model.obtain_origin_stat(clean_data_loader)

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    evaluate_tta(loader, tta_model, student, f'RoTTA+FOA({model_name})-cifar100', attack_type)

def main():
    models_path = '../Training/Models'
    dataset = f"../Dataset/cifar-100/test"
    models = {'ResNet18' : 'trained_resnet.pth'}  
    attacks = ['FGSM', 'PGD', 'CW','AutoAttack']
    student = torch.load(f'{models_path}/{models['ResNet18']}')
    for model_name in models :
        # for attack in attacks :
            student = torch.load(f'{models_path}/{models[model_name]}')
            dataset_dir = f"{dataset}"
            print(dataset_dir)
            testTimeAdaptation(student, dataset_dir, model_name, "Clean")

if __name__ == "__main__":
    main()