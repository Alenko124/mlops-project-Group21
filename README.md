# Project Description

This project will address a real-world **image classification** problem using satellite imagery. The objective will be to classify land-use types from remote sensing images using transfer learning with a pretrained convolutional neural network.

---

## Dataset

The project will use the **EuroSAT** dataset, accessed via the Hugging Face Hub:

- **Dataset source:** https://huggingface.co/datasets/nielsr/eurosat-demo  
- **Data type:** RGB satellite images  
- **Task:** Multi-class image classification  
- **Number of classes:** 10 land-use categories  

The dataset will contain 27,000 RGB satellite images with a resolution of 64 × 64 pixels, representing different land-use classes such as forests, residential areas, rivers, highways, and industrial zones. Given an input satellite image, the model will predict the corresponding land-use class.

---

## Model and Training Strategy

The task will be solved using **transfer learning**. A pretrained convolutional neural network will be fine-tuned on the EuroSAT dataset. As the baseline architecture, **ResNet-18** will be used (subject to change), initialized with ImageNet-pretrained weights obtained from the `timm` model repository:

- **Pretrained model source:** https://huggingface.co/timm/resnet18.a1_in1k

---

## Data Augmentation

To improve generalization, data augmentation will be applied during training using the **Albumentations** library. The augmentation pipeline will include random cropping, horizontal flipping, rotation, and photometric transformations.

---

## Frameworks and Tools

The project will be implemented using the following frameworks and tools:

- PyTorch for model implementation and training  
- Hugging Face `datasets` for dataset loading  
- `timm` for pretrained image classification models  
- Albumentations for data augmentation  
- GitHub for collaboration  
- Docker for reproducible environments  
- Experiment monitoring for tracking configurations and performance  
- ...

---

## Course Features

All experiments will be executed in a reproducible environment. A Docker container will be used to ensure consistent dependencies across systems, and the `uv` package manager will be used for dependency and environment management. Training configurations and experimental results will be logged to enable reproducibility and systematic comparison. In addition, the project will incorporate relevant features and techniques introduced during the course as they are learned and become applicable.
