import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from helper import evaluate_tta
import timm

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

def testTimeAdaptation(student, dataset_path, attack_type):
    batch_size = 32
    # model, optimizer
    model = student
    params = model.parameters()

    optimizer = build_optimizer()

    tta_model = RoTTA(model, optimizer)
    tta_model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = "../Dataset/tiny/CIFAR-10/test"
    dataset = datasets.ImageFolder(train_data, transform=transform)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    tta_model.obtain_origin_stat(train_loader)

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


    evaluate_tta(loader, tta_model, 'MVN2', attack_type)

def main():
    attacks = ['FGSM', 'PGD', 'CW', 'AutoAttack']
    model_name = 'ResNet18'
    # teacher = timm.create_model('wide_resnet50_2', pretrained=True)
    # teacher.fc = nn.Linear(teacher.fc.in_features, 10)
    # for attack_type in attacks :
    student = torch.load('../Training/Models/trained_mobilenetv2.pth')
    dataset_dir = f"../Dataset/tiny/CIFAR-10/test"
    testTimeAdaptation(student, dataset_dir, 'Clean')

if __name__ == "__main__":
    main()