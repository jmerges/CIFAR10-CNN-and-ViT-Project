import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms, models
from torch.utils.tensorboard import SummaryWriter
import math

batch_size = 64
epochs = 5
warmup_epochs = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Selected device:", device)
print("torch:", torch.__version__, "torch.cuda.build:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    print("device name (0):", torch.cuda.get_device_name(0))

writer = SummaryWriter(log_dir="runs/vit_tiny_cifar10")

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

train_dataset = datasets.CIFAR10('./data', train=True, transform=train_transform)
test_dataset = datasets.CIFAR10('./data', train=False, transform = test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1, dropout=0.2)

model.heads.head = nn.Linear(model.heads.head.in_features, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

total_steps = epochs * len(train_loader)
warmup_steps = warmup_epochs * len(train_loader)

def cosine_scheduler(step):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

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
            total += len(targets)

    return total_loss / len(loader), 100 * correct / total

# Train
model.train()
step = 0
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

        # Update LR with cosine schedule
        lr = 3e-4 * cosine_scheduler(step)
        for p in optimizer.param_groups:
            p["lr"] = lr

        if batch_idx % 1 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item()}")
    
    train_loss, train_acc = evaluate(model, train_loader)
    val_loss, val_acc = evaluate(model, test_loader)

    writer.add_scalar("Loss/train_epoch", train_loss, epoch)
    writer.add_scalar("Loss/val_epoch", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("LR", lr, epoch)

    print(f"Epoch {epoch+1}/{epochs} "
        f"| Train Loss {train_loss:.4f} Acc {train_acc:.2f}% "
        f"| Val Loss {val_loss:.4f} Acc {val_acc:.2f}% "
        f"| LR {lr:.6f}")

writer.close()
print("Training done!")
