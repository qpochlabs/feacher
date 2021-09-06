<p align="center">
    <img src="https://github.com/qpochlabs/feacher/blob/main/assets/logo.png" width="85" alt="feacher-logo"/>
</p>

<p align="center">
    <img src="https://github.com/qpochlabs/feacher/workflows/Python%20package/badge.svg" alt="Python-Package"/>
    <img src="https://badge.fury.io/py/feacher.svg" alt="PyPi"/>
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT-Licence"/>
    <img src="https://pepy.tech/badge/feacher/month" alt="PePy-Downloads"/>
</p>

# Feacher
Feacher is a light-weight Image feature extraction library that can help in transfer learning applications.

## Requirements
-   PyTorch
-   Torchvision
-   PIL

## Installation
```python
pip install feacher
```

## Usage
```python
import feacher

# Path/Directory containg images 
path_to_images = 'images/'

# Extract features and add them to a list
image_features = feacher.extract(path_to_images,
                                 pretrained_model='resnet18',
                                 layer='avgpool',
                                 layer_size=512,
                                 resize_dim=256)

print(image_features[0])
```