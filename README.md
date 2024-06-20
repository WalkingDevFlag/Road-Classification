# Road Classification Model 

## Overview
This repository contains a PyTorch Lightning-based implementation of a road classification model using the U-Net architecture. The model is trained on a dataset of images with corresponding masks, and it is designed to predict the road class in each pixel of the input image.

## Requirements
- **PyTorch**: Version 1.12.1 or higher.
- **PyTorch Lightning**: Version 1.6.1 or higher.
- **Albumentations**: Version 1.2.0 or higher.
- **OpenCV**: Version 4.5.5.64 or higher.
- **Pandas**: Version 1.4.4 or higher.
- **Matplotlib**: Version 3.5.2 or higher.
  
## Dataset

The dataset consists of satellite images and their corresponding segmentation masks, organized into images and mask directories. The class definitions are stored in a 'class_dict.csv' file.

I've used the DeepGlobe Land-Cover Dataset. I separated the images in the `train` folder into 2 folders, `Images` and `Masks`

Link to the Dataset -> [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)

## Model Architecture
The model consists of several layers, including:
- **UnetDecoder**: A U-Net decoder with two stages and four feature channels.
- **SegmentationHead**: A segmentation head with two convolutional layers and an activation function.

## Training and Testing
The model is trained using the Adam optimizer and a batch size of 8. The training process is monitored using early stopping and model checkpointing. The model is tested on a separate test dataset, and the test metrics include accuracy, F1 score, intersection over union (IOU), precision, and recall.

## Model Evaluation
The model's performance is evaluated using the following metrics:
- **Accuracy**: The proportion of correctly classified pixels.
- **F1 Score**: The harmonic mean of precision and recall.
- **Intersection Over Union (IOU)**: The ratio of the intersection to the union of the predicted and true masks.
- **Precision**: The proportion of true positives among all predicted positives.
- **Recall**: The proportion of true positives among all actual positives.

## Usage
1. Clone the repository and navigate to the directory.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the training script using `python train.py`.
4. Load the trained model and test it using `python test.py`.

## Liscense
Lmao no Liscense


