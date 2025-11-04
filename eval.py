import torch

def test(model,test_loader,device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
        
            outputs,_ = model(data)
            _, predicted = torch.max(outputs, 1)
            
            total += len(labels)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100*correct/total:.2f}%")
    return correct/total