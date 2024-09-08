import os
import time
import random
from importlib import reload, import_module

import torch    
from torch.utils.data import DataLoader
import timm
import numpy as np

from torchvision import datasets, transforms

from foa import FOA
from prompt import PromptViT
from cli_utils import *
from metrics import ECELoss

def validate_adapt(val_loader, model):
    print("-" * 30)
    print("Within the validate_adapt() function")
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    
    outputs_list, targets_list = [], []
    with torch.no_grad():
        end = time.time()
        for batch_idx, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()
            print("Predicting and Updating the Model using FOA for the current batch")
            output = model(images)

            # for calculating Expected Calibration Error (ECE)
            outputs_list.append(output.cpu())
            targets_list.append(target.cpu())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            del output

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 5 == 0:
                print(progress.display(batch_idx))
            
        outputs_list = torch.cat(outputs_list, dim=0).numpy()
        targets_list = torch.cat(targets_list, dim=0).numpy()
        
        logits = True 
        ece_avg = ECELoss().loss(outputs_list, targets_list, logits=logits) # calculate ECE
    print(f"Exiting the validate_adapt() function")
    print("-" * 30)
    return top1.avg, top5.avg, ece_avg

def obtain_loader(dataset_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return train_loader


if __name__ == '__main__':

    print("-" * 30)
    print("Within the main Function")
    seed = 42
    quant = False
    train_dataset_path = '../Dataset/CIFAR-10/train'
    test_dataset_path = '../Dataset/CIFAR-10/test'
    attack = 'PGD'
    # set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    print("Loading the base Model : vit_base_patch16_224")
    net = timm.create_model('vit_base_patch16_224', pretrained=True)
        
    net = net.cuda()
    net.eval()
    net.requires_grad_(False)

    print("Creating the Prompt ViT Model with 3 Prompts")
    net = PromptViT(net, 3).cuda()
    print(f"Setting up the Adaptation Algorithm FOA")
    adapt_model = FOA(net, 0.4)
    train_loader = obtain_loader(train_dataset_path)
    print(f"Calculating the Source Statistics")
    adapt_model.obtain_origin_stat(train_loader)
    adapt_model.imagenet_mask = None

    acc_list, ece_list = [], []

    val_loader = obtain_loader(test_dataset_path)
    torch.cuda.empty_cache()

    print("Calling the validate_adapt function()")
    top1, top5, ece_loss = validate_adapt(val_loader, adapt_model)
    print(f"Under Attack type {attack} After FOA Top-1 Accuracy: {top1:.6f} and Top-5 Accuracy: {top5:.6f} and ECE: {ece_loss:.6f}")
    acc_list.append(top1)
    ece_list.append(ece_loss)

    mean_acc = sum(acc_list) / len(acc_list)
    mean_ece = sum(ece_list) / len(ece_list)
    print(f"Final Accuracy : {mean_acc} and ECE Loss : {mean_ece}")
    print("Exiting the Code")
    print("-" * 30)