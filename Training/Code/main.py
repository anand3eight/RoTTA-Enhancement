import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from helper import evaluate_tta
from rotta import RoTTA
from WRN import *
import argparse

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

    tta_model = RoTTA(student, teacher, optimizer, attack_type)
    tta_model.cuda()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    evaluate_tta(loader, tta_model, args.desc, attack_type, args.arch)

def get_args() :
    parser = argparse.ArgumentParser(description='RADiC Defense')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--optimizer', default='Adam', type=str, help='Optimizer')
    parser.add_argument('--dataset', default='CIFAR-10', type=str, help='Dataset')
    parser.add_argument('--model', default='resnet-18', type=str, help='Base Model')
    parser.add_argument('--kd', default=False, type=bool, help='Knowledge Distillation')
    parser.add_argument('--teacher', default='wrn34-10', type=str, help='Teacher Model')
    parser.add_argument('--student', default='resnet-18', type=str, help='Student Model')
    parser.add_argument('--adaptive_attack', default=False, type=bool, help='Adaptive Attacks')
    parser.add_argument('--arch', default='radic', type=str, help='Architecture')
    parser.add_argument('--desc', default='Log', type=str, help='Description')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    dataset_dir = f"../Dataset/tiny/{args.dataset}/test"
    student_models = {'resnet-18' : 'trained_resnet.pth', 'mobilenet-v2' : 'trained_mobilenetv2.pth'}
    teacher_models = {'wrn34_10' : 'trained_wide_resnet34_10.pth', 'wrn34_20' : 'trained_wide_resnet34_20.pth'}
    teacher = None

    if args.adaptive_attack == True : 
        attacks = ['CTA', 'MBA', 'EOTA']
    else : 
        attacks = ['FGSM', 'PGD', 'CW', 'APGD', 'APGD-t', 'FAB-t', 'Square', 'AA', 'JSMA']

    if args.kd == True :
        teacher_path = teacher_models[args.teacher]
        student_path = student_models[args.student]
        teacher = torch.load(f'{models_path}/{teacher_path}')
        student = torch.load(f'{models_path}/{student_path}')
    else : 
        student = f'../Training/Models/{args.model}'

    for attack in attacks :
        testTimeAdaptation(student, teacher, dataset_dir, attack, args)

if __name__ == '__main__' : 
    main()
