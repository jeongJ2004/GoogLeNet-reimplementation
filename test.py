import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from model.googlenet import GoogLeNet
import torchvision.transforms as transforms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    test_dataset = CIFAR10(root="/home/jaeyi/code/vggnet-reimplementation/data/datasets/cifar-10", train=False, download=False, transform=test_transforms)

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = GoogLeNet(10).to(device)
    model.load_state_dict(torch.load("./checkpoints/best_googlenet.pth", map_location=device))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            preds = outputs.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

    print(f"Test Acc : {100 * (correct/total):.2f} %")


if __name__ == "__main__":
    main()
