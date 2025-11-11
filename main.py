import torch
import torch.nn as nn
from attention import RealViT,RealViTconv
from c_attention import ComplexViT,initialise_complex_weights,ComplexViTconv
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
from train import train
from eval import test
import numpy as np

class ToComplex64:
    def __call__(self, tensor):
        return tensor.to(torch.complex64)


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #hyperparams
    image_size = 32
    embed_dim = 256
    num_heads = 8
    num_blocks = 6
    num_classes = 10
    patch_size = 4
    encoder_ff_hidden_dim = embed_dim*4
    train_epochs = 50
    lr= 3e-4
    weight_decay = 0

    # model = RealViT(image_size,embed_dim=embed_dim,num_heads=num_heads,num_blocks=num_blocks,num_classes=num_classes,patch_size=patch_size,encoder_ff_hidden_dim=encoder_ff_hidden_dim).to(device)
    model = ComplexViT(image_size,embed_dim=embed_dim,num_heads=num_heads,num_blocks=num_blocks,num_classes=num_classes,patch_size=patch_size,encoder_ff_hidden_dim=encoder_ff_hidden_dim).to(device)
    # model = ComplexViTconv(image_size,embed_dim=embed_dim,num_heads=num_heads,num_blocks=num_blocks,num_classes=num_classes,patch_size=patch_size,encoder_ff_hidden_dim=encoder_ff_hidden_dim).to(device)
    # model = RealViTconv(image_size,embed_dim=embed_dim,num_heads=num_heads,num_blocks=num_blocks,num_classes=num_classes,patch_size=patch_size,encoder_ff_hidden_dim=encoder_ff_hidden_dim).to(device)

    initialise_complex_weights(model,device)

    train_transform_list = [
    transforms.RandomCrop(image_size, padding=4), 
    transforms.RandomHorizontalFlip(),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2464, 0.2428, 0.2607]),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)), 
    ]

    if model.__class__.__name__ == "ComplexViT":
        train_transform_list.append(ToComplex64())

    train_transform = transforms.Compose(train_transform_list)

    print(f"Model: {model.__class__.__name__}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Patch size: {patch_size}")
    print(f"Embed Dim: {embed_dim}")
    print(f"Transformer blocks: {num_blocks}")
    print(f"Attention heads: {num_heads}")
    print(f"FF hidden dim: {encoder_ff_hidden_dim}")
    print(f"Num classes: {num_classes}")
    print(f"Learning rate: {lr}")
    print(f"Weight decay: {weight_decay}")

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,num_workers=4)

    #lr scheduler?
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss() 

    test_transform_list = [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2464, 0.2428, 0.2607]),
    ]

    if model.__class__.__name__ == "ComplexViT":
        test_transform_list.append(ToComplex64())

    test_transform = transforms.Compose(test_transform_list)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,num_workers=4)

    model_type_str = model.__class__.__name__

    model_string = (
        f"{model_type_str}"
        f"_ed{embed_dim}"
        f"_fc{encoder_ff_hidden_dim}"
        f"_ps{patch_size}"
        f"_{train_epochs}epochs"
        f"_{lr}lr"
        f"_{weight_decay}wd"
    )

    print(f"Model string: {model_string}")

    train_accuracies,train_losses,test_accuracies,x_steps,epoch_times = train(model,optimizer,train_loader,device,loss_fn,num_epochs=train_epochs,savepath="diss_models/"+model_string+".pth",test_loader=test_loader)

    print("train_accuracies:", train_accuracies)
    print("train_losses:", train_losses)
    print("test_accuracies:", test_accuracies)
    print("epoch_times:", epoch_times)

    test(model,test_loader,device)
