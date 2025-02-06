import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from helper import evaluate_tta
from rotta import RoTTA
from WRN import *

def build_optimizer(method = 'Adam'):
    def optimizer(params):
        if method == 'Adam':
            return torch.optim.Adam(params, lr=1e-3)
        
        elif method == 'SGD':
            return torch.optim.SGD(params, lr=1e-3)

        else:
            raise NotImplementedError

    return optimizer

def testTimeAdaptation(student, teacher, dataset_path, attack_type):
    
    batch_size = 4
    # model, optimizer

    optimizer = build_optimizer()

    tta_model = RoTTA(student, teacher, optimizer, attack_type)
    tta_model.cuda()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    evaluate_tta(loader, tta_model, 'WithoutUpdates', attack_type)

def main():
    models_path = '../Training/Models'
    dataset_dir = f"../Dataset/tiny/CIFAR-10/test"
    student_models = {'Resnet18' : 'trained_resnet.pth', 'MobileNetV2' : 'trained_mobilenetv2.pth'}
    teacher_models = {'WRN34_10' : 'trained_wide_resnet34_10.pth', 'WRN34_20' : 'trained_wide_resnet34_20.pt'}
    attacks = ['FGSM']
    for teacher_path in teacher_models : 
        for student_path in student_models :
            for attack in attacks :
                student = torch.load(f'{models_path}/{student_models[student_path]}')
                teacher = torch.load(f'{models_path}/{teacher_models[teacher_path]}')
                testTimeAdaptation(student, teacher, dataset_dir, attack)

if __name__ == "__main__":
    main()