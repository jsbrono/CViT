import torch
import torch.nn as nn
import time
from eval import test

def train(model:nn.Module,optimizer:torch.optim.Adam,train_loader,device:torch.device,loss_fn:nn.Module,num_epochs:int,savepath:str|None,test_loader:str):
    #Performs a training loop over num_epochs for a given model

    train_accuracies = []
    test_accuracies = []
    x_steps = []
    train_losses = []
    epoch_times = []

    acc = test(model,test_loader,device)
    test_accuracies.append(acc)
    model.train()

    for epoch in range(num_epochs):
        correct = 0
        total = 0
        start_time = time.time()
        
        for i, (data,labels) in enumerate(train_loader):

            data = data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs,_ = model(data)
            
            loss_fn.to(device)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs, 1)
            total += len(labels)

            correct += (predicted==labels).sum().item()
            train_losses.append(loss.item())

        epoch_times.append(time.time()-start_time)
        print(f"Epoch: {epoch+1}/{num_epochs} completed in {time.time()-start_time:.2f} seconds")
        print(f"Epoch: {epoch+1}/{num_epochs} Train Accuracy: {100*correct/total:.2f}%")

        train_accuracies.append(correct/total)
        acc=test(model,test_loader,device)
        test_accuracies.append(acc)
        model.train()
        x_steps.append(epoch+1)

    if savepath is not None:
        torch.save(model.state_dict(),savepath)
        print(f"model saved at {savepath}")

    return train_accuracies,train_losses,test_accuracies,x_steps,epoch_times
