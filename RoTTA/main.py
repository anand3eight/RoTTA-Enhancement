import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from helper import evaluate_tta
from rotta import RoTTA
from WRN import WideResNet, NetworkBlock, BasicBlock

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

    tta_model = RoTTA(student, teacher, optimizer)
    tta_model.cuda()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    evaluate_tta(loader, tta_model, 'RoTTA+ADaaD', attack_type)

def main():
    # attacks = ['FGSM', 'PGD', 'CW', 'AutoAttack']
    # model_name = 'ResNet18'
    # teacher = timm.create_model('wide_resnet50_2', pretrained=True)
    # teacher.fc = nn.Linear(teacher.fc.in_features, 10)
    # for attack_type in attacks :
    student = torch.load('../Training/Models/trained_resnet.pth')
    teacher = torch.load('../Training/Models/trained_wide_resnet34_10.pth')
    dataset_dir = f"../Dataset/tiny/CIFAR-10/test"
    testTimeAdaptation(student, teacher, dataset_dir, 'PGD')

if __name__ == "__main__":
    main()