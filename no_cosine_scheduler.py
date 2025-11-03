from google.colab import drive
drive.mount('/content/drive')
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

batch_size = 64
epochs = 10
lr=3e-5
batch_losses = []
batch_epochs = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected device:", device)
print("torch:", torch.__version__, "torch.cuda.build:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    print("device name (0):", torch.cuda.get_device_name(0))

writer = SummaryWriter(log_dir="runs/vit_tiny_cifar10")

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

train_dataset = datasets.CIFAR10('./data', train=True, transform=transform_train, download=True)
test_dataset = datasets.CIFAR10('./data', train=False, transform = transform_test, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1, dropout=0.2)

model.heads.head = nn.Linear(model.heads.head.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)

# Test
def evaluate(model, loader):
    model.eval()
    total, correct, total_loss = 0, 0, 0
    
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# Train
model.train()
for epoch in range(epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        # Forward Pass
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Back Prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        batch_epochs.append(epoch + batch_idx/len(train_loader))

        if batch_idx % 1 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")

    train_loss, train_acc = evaluate(model, train_loader)
    test_loss, test_acc = evaluate(model, test_loader)

    writer.add_scalar("Loss/train_epoch", train_loss, epoch)
    writer.add_scalar("Loss/val_epoch", test_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", test_acc, epoch)
    writer.add_scalar("LR", lr, epoch)

train_loss, train_acc = evaluate(model, train_loader)
test_loss, test_acc = evaluate(model, test_loader)

print(f"\nResults:")
print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
print(f"Val   Loss: {test_loss:.4f} | Val   Acc: {test_acc:.2f}%")

plt.figure(figsize=(10,5))
plt.plot(batch_epochs, batch_losses, label="Batch Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Batch")
plt.grid(True)
plt.show()
