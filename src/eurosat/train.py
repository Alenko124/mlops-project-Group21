import torch
import torch.nn as nn
import torch.optim as optim
from data import create_dataloaders
from model import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, _ = create_dataloaders(batch_size=32, num_workers=4)

model = create_model(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

epochs = 1

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
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
        train_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    train_loss /= total
    train_acc = correct / total

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    val_loss /= total
    val_acc = correct / total

    print(f"Epoch {epoch+1}/{epochs} | TrainLoss {train_loss:.4f} TrainAcc {train_acc:.4f} | ValLoss {val_loss:.4f} ValAcc {val_acc:.4f}")

torch.save(model.state_dict(), "models/resnet18_eurosat.pt")
