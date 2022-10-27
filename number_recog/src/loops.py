import torch

def train_step(model, batch, criterion, device):
    x, y = batch
    x, y = x.squeeze().to(device), y.to(device)

    pred = model(x)
    loss = criterion(pred, y)
    return loss

def valid(dataloader, model, device):
    with torch.no_grad():
        # loop = tqdm(dataloader)
        num_right = 0
        num_sample = 0
        for i, (x, y) in enumerate(dataloader):
            num_sample += len(y)
            x, y = x.to(device), y.to(device)
            pred = model(x)
            pred = torch.argmax(pred, dim=1)
            num_right += torch.sum(pred == y)
        return num_right / num_sample