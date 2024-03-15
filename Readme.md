# Multiclass Semantic Segmentation on Cityscapes Dataset

This repository contains an implementation of multiclass semantic segmentation on the Cityscapes dataset using deep learning, particularly employing the UNet architecture. Additionally, the dataset has undergone preprocessing to consolidate it into 8 classes deemed more significant for the task.
The mainly part of the project is described in the MainNotebook.ipynb file, other files like UNet.py and CityscapesLoader.py are imported in the main file

## Table of Contents
- [Introduction](#introduction)
- [Python Version](#python-version)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Architecture](#architecture)
- [Training](#training)
- [Optimizers and Loss Function](#optimizers-and-loss-function)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Introduction
Semantic segmentation is a crucial task in computer vision that involves assigning semantic labels to each pixel in an image. This project focuses on performing multiclass semantic segmentation using deep learning techniques, specifically leveraging the UNet architecture. The Cityscapes dataset, a popular benchmark for urban scene understanding, is employed for training and evaluation.

## Python Version
This project is implemented using Python 3.10.10. It's recommended to use Python 3.10.10 or check the libraries available in the chosen Python version.

## Dataset
The Cityscapes dataset consists of urban street scenes, with high-quality pixel-level annotations. It contains images from various cities, captured under different weather and lighting conditions. The dataset includes 30 classes for segmentation tasks. However, for the purpose of this project, we have preprocessed the dataset to reduce it to 8 classes that are more relevant for our segmentation task.
The version of the Cityscapes dataset used in this project can be downloaded from the following link: [Cityscapes Dataset](https://www.kaggle.com/datasets/xiaose/cityscapes/code). It's important to ensure that you're using the correct version of the dataset to reproduce the results and maintain compatibility with the preprocessing steps implemented in this project. If you encounter any issues with the dataset version or require assistance, feel free to reach out for support.

## Preprocessing
Prior to training, the Cityscapes dataset underwent preprocessing to reduce the number of classes to 8. This step was taken to focus on the most important classes for semantic segmentation while simplifying the task. The preprocessing step involved merging similar classes and removing less significant ones.

## Architecture
The UNet architecture is employed for semantic segmentation tasks due to its effectiveness in capturing spatial information while maintaining high-resolution features. UNet consists of an encoder-decoder structure with skip connections, allowing for precise localization of objects in images.

## Training
The model is trained using the preprocessed Cityscapes dataset with a focus on the 8 selected classes. The training process involves optimizing the model parameters to minimize a specified loss function while utilizing techniques such as data augmentation to improve generalization. 

## Optimizers and Loss Function
During training, the AdamW optimizer is utilized for parameter optimization. Additionally, a Cosine Annealing learning rate scheduler is employed to adjust the learning rate throughout the training process. The loss function used is a Weighted Cross Entropy Loss, which assigns different weights to each class to account for class imbalance.

## Results
The performance of the trained model is evaluated using standard metrics such as mean IoU (Intersection over Union).

## Usage
To use this project, follow these steps:
1. Clone the repository.
2. Install the required dependencies.
3. Preprocess the Cityscapes dataset according to your preferences (optional).
4. Train the model using the provided notebook.
5. Evaluate the trained model using the provided evaluation tools.

## License
This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code for your own purposes.
