<div style="margin:0 auto;"><img src="https://github.com/qpochlabs/feacher/blob/main/assets/logo.png" width="85"/></div>
<br><br>
[![Build Status](https://github.com/qpochlabs/feacher/workflows/Python%20package/badge.svg)](https://github.com/qpochlabs/feacher/actions)
[![PyPI version](https://badge.fury.io/py/feacher.svg)](https://pypi.org/project/feacher/)
[![License MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/qpochlabs/feacher/blob/master/LICENSE)
<!-- [![Downloads](https://pepy.tech/badge/feacher/month)](https://pepy.tech/project/feacher) -->

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