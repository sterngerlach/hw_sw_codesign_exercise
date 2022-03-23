# coding: utf-8
# train.py

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets
import torchvision.transforms

from toynet import ToyNet

def train(model: ToyNet,
          device: torch.device,
          train_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          epoch: int):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print("Train epoch: {} ({} / {}, {:.0f}%), loss: {:.6f}".format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100.0 * batch_idx / len(train_loader), loss.item()))

def test(model: ToyNet,
         device: torch.device,
         test_loader: torch.utils.data.DataLoader,
         epoch: int):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            test_loss += F.nll_loss(out, target, reduction="sum").item()
            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print("Test epoch: {}, loss: {:.4f}, accuracy: {} / {} ({:.0f}%)".format(
          epoch, test_loss, correct, len(test_loader.dataset),
          100.0 * correct / len(test_loader.dataset)))

def main():
    # Load the dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    train_set = torchvision.datasets.MNIST(
        "./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=32, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=32, shuffle=False, num_workers=1)

    # Create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ToyNet().to(device)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, 6):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch)

    # Save the model
    torch.save(model.state_dict(), "toynet.pth")

if __name__ == "__main__":
    main()
