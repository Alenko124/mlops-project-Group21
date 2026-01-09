import torch
import torch.nn as nn
import timm


NUM_CLASSES = 10


def create_model(device=None):
    """
    Creates a pretrained ResNet-18 model and replaces the
    fully connected layer for EuroSAT (10 classes).
    Backbone is frozen, only FC layer is trainable.
    """

    # Load pretrained ResNet-18
    model = timm.create_model(
        "resnet18.a1_in1k",
        pretrained=True,
    )

    # Replace fully connected layer
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES,
    )

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    # Move to device if provided
    if device is not None:
        model = model.to(device)

    return model


# --------------------------------------------------
# Sanity check (optional)
# --------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device)

    print(model)
    print("\nTrainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("  ", name, param.shape)
