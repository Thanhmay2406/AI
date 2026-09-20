import torch
import torch.nn as nn
from network import ResNet18
from torchvision import transforms, datasets
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

print(f"Device: {device}")

train_transform = transforms.Compose([

    # Data augmentation
    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    # CIFAR-10 mean/std
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])


test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

model = ResNet18(10)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                      weight_decay=5e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

def train_one_epoch():
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size[0]
        _, predicted = outputs.max(1)
        total += labels.size[0]
        correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size[0]
        _, predicted = outputs.max(1)
        total += labels.size[0]
        correct += (predicted == labels).sum().item()
    avg_loss = total_loss / total
    accuracy = correct / total * 100
    return avg_loss, accuracy

num_epochs = 100
best_accuracy = 0.0

for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch()
    test_loss, test_acc = evaluate()
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(
        f"Epoch [{epoch + 1:03d}/{num_epochs}] "
        f"LR: {current_lr:.6f} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Test Loss: {test_loss:.4f} | "
        f"Test Acc: {test_acc:.2f}%"
    )
    if test_acc > best_accuracy:
        best_accuracy = test_acc
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "accuracy":
                    test_acc
            },
            "best_resnet18_cifar10.pth"
        )
        print(
            f"Saved best model: "
            f"{best_accuracy:.2f}%"
        )
    print("done!")

print(
    f"Best test accuracy: "
    f"{best_accuracy:.2f}%"
)
