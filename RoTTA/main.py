import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from torchvision.models import ResNet18_Weights

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

def testTimeAdaptation(base_model, dataset_path):
    print("-" * 30)
    print("In the TTA Function")

    batch_size = 32
    # model, optimizer
    model = base_model
    params = model.parameters()

    optimizer = build_optimizer()

    tta_model = RoTTA(model, optimizer)
    tta_model.cuda()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset Loaded with Batch Size = {batch_size}")
    correct_predictions = 0
    total_predictions = 0

    tbar = tqdm(loader)
    for batch_id, (images, labels) in enumerate(tbar):
        images, labels = images.cuda(), labels.cuda()
        print(f"Calling the RoTTA Module")
        output = tta_model(images)
        predict = torch.argmax(output, dim=1)
        accurate = (predict == labels).sum().item()
        correct_predictions += accurate
        total_predictions += labels.size(0)    
        current_accuracy = correct_predictions / total_predictions
        tbar.set_postfix( accuracy = current_accuracy ) 

    final_accuracy = correct_predictions / total_predictions
    print(f"Final Accuracy : {final_accuracy}")
    print(f"Exiting the TTA Function after calculation of Final Accuracy")
    print("-" * 30)

def main():
    print("-" * 30)
    print("Setting up the Base Model : Resnet-18 and PGD attacked CIFAR-10 Dataset")
    base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_classes = 10
    base_model.fc = torch.nn.Linear(base_model.fc.in_features, num_classes)
    dataset_path = '../Attacks/PGD/CIFAR-10/ViT'
    print("Calling the TTA function")
    testTimeAdaptation(base_model, dataset_path)
    print(f"Exiting the Main Function")
    print("-" * 30)

if __name__ == "__main__":
    main()