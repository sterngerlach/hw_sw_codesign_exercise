# coding: utf-8
# test.py

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets
import torchvision.transforms

from toynet import ToyNet

def test(model: ToyNet,
         device: torch.device,
         test_loader: torch.utils.data.DataLoader):
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

    print("Test loss: {:.4f}, accuracy: {} / {} ({:.0f}%)".format(
          test_loss, correct, len(test_loader.dataset),
          100.0 * correct / len(test_loader.dataset)))

def main():
    # Load the dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    test_set = torchvision.datasets.MNIST(
        "./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=32, shuffle=False, num_workers=1)

    # Create the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ToyNet().to(device)
    model.load_state_dict(torch.load("toynet.pth"))

    # Test the model
    test(model, device, test_loader)

if __name__ == "__main__":
    main()
