import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

from RADiC.modules.helper import evaluate_tta
from RADiC.defense import RoTTA
from Models.networks.resnet import ResNet18

def build_optimizer(method = 'Adam'):
    def optimizer(params):
        if method == 'Adam':
            return torch.optim.Adam(params, lr=1e-3)

        elif method == 'SGD':
            return torch.optim.SGD(params, lr=1e-3)

        else:
            raise NotImplementedError

    return optimizer

def testTimeAdaptation(student, teacher, dataset_path, attack_type, args):

    optimizer = build_optimizer(args.optimizer)

    tta_model = RoTTA(student, teacher, optimizer, attack_type, args)
    tta_model.cuda()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    evaluate_tta(loader, tta_model, student, args.desc, attack_type)

def get_args() :
    parser = argparse.ArgumentParser(description='RADiC Defense')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--optimizer', default='Adam', type=str, help='Optimizer')
    parser.add_argument('--dataset', default='CIFAR-10', type=str, help='Dataset')
    parser.add_argument('--kd', default=False, type=bool, help='Knowledge Distillation')
    parser.add_argument('--teacher', default='wrn34_10', type=str, help='Teacher Model')
    parser.add_argument('--student', default='resnet18', type=str, help='Student Model')
    parser.add_argument('--adaptive_attack', default=False, type=bool, help='Adaptive Attacks')
    parser.add_argument('--arch', default='rotta', type=str, help='Architecture')
    parser.add_argument('--desc', default='Log', type=str, help='Description')
    args = parser.parse_args()
    return args

def load_models(args, models_path) :

    student_models = {'resnet18' : 'resnet18.pth', 'mobilenetv2' : 'mobilenetv2.pth'}
    teacher_models = {'wrn34_10' : 'wide_resnet34_10.pth', 'wrn34_20' : 'wide_resnet34_20.pth'}

    teacher = None
    if args.kd == True :
        teacher_path = teacher_models[args.teacher]
        teacher = torch.load(f'{models_path}/{teacher_path}')

    student_path = student_models[args.student]
    student = ResNet18(num_classes=10)
    student.load_state_dict(torch.load(f'{models_path}/{student_path}'))

    return student, teacher

def main():
    args = get_args()
    base_path = "/home/project/Documents/RoTTA-Enhancement/Sem8/"
    models_path = f"{base_path}/Models/saved_models"
    dataset_dir = f"{base_path}/Dataset/{args.dataset}/test"
    teacher = None

    if args.adaptive_attack == True :
        attacks = ['CTA', 'MBA', 'EOTA']
    else :
        attacks = ['Square', 'AA', 'JSMA']

    for attack in attacks :
        student, teacher = load_models(args, models_path)
        print(f"Attack : {attack}")
        testTimeAdaptation(student, teacher, dataset_dir, attack, args)

if __name__ == "__main__" :
    main()