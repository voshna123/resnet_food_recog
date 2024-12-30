import torch
import torchvision

def create_model(classes: int = 3):
    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights = weights)

    for params in model.parameters():
        params.requires_grad= False
    
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(512, 101)
    )

    transform = weights.transforms()

    return model, transform

