import torch
from data import create_dataloaders
from model import create_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataloader
_, _, test_loader = create_dataloaders(batch_size=32, num_workers=4)

# Load model
model = create_model(device)
model.load_state_dict(torch.load("models/resnet18_eurosat.pt", map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test accuracy: {accuracy:.4f}")
