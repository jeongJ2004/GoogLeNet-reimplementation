import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from tqdm import tqdm

from model.googlenet import GoogLeNet  

# create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

# data root
data_root = "/home/jaeyi/code/vggnet-reimplementation/data/datasets/cifar-10"

# transforms (inline, no utils file)
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

# datasets and loaders
batch_size = 128
train_ds = CIFAR10(root=data_root, train=True, download=False, transform=train_transform)
val_ds   = CIFAR10(root=data_root, train=False, download=False, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

# device, model, loss, optimizer, scheduler
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = GoogLeNet(10, True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

best_val_acc = 0.0
epochs = 50

for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
        imgs, lbls = imgs.to(device), lbls.to(device)
        outputs = model(imgs)
        # Now it is different from other models that I've reimplemented, because this model has to handle aux outputs
        if isinstance(outputs, tuple):
            main_out, aux1_out, aux2_out = outputs
            loss_main = criterion(main_out, lbls)
            loss_aux1 = criterion(aux1_out, lbls)
            loss_aux2 = criterion(aux2_out, lbls)
            loss = loss_main + 0.3 * (loss_aux1 + loss_aux2) # mentioned in the paper : "(the losses of the auxiliary classifiers were weighted by 0.3"
        else :
            loss = criterion(outputs, lbls)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{epochs} -> Train Loss : {avg_train_loss:4f}")


    # Time for Validation set
    with torch.no_grad():
        correct = 0
        total = 0
        for imgs, lbls in tqdm(val_loader, desc="Validation", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, preds = outputs.max(1)
            total += lbls.size(0)
            correct += (preds == lbls).sum().item()

        val_acc = correct / total
        print(f"Validation Acc : {100 * val_acc:.2f} %")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_googlenet.pth")
            print(f"Saved new Best model (Val acc : {100 * best_val_acc:.2f}) %")

    scheduler.step()
