# COVID-19 Radiography Detection

This project provides a robust deep learning pipeline using **PyTorch** to classify chest X-ray images into four different categories: `COVID`, `Normal`, `Viral Pneumonia`, and `Lung_Opacity`. It leverages modern convolutional architectures, namely **EfficientNet-B0** and **ConvNeXt-Tiny**, to evaluate their comparative effectiveness in medical image diagnostics. 

## Project Overview

- **Problem Definition:** Automatically detect signs of COVID-19, Viral Pneumonia, or Lung Opacity from Chest X-Rays.
- **Dataset:** COVID-19 Radiography Database.
- **Deep Learning Framework:** PyTorch & `timm` (PyTorch Image Models).

## Key Features

- **Data Preprocessing & Augmentation**:
  - Resizing images to 256x256.
  - Adding robustness through synthetic variation: Random Rotations, Horizontal Flips, and Color Jitter.
  - Using ImageNet standards for RGB normalization.
- **Model Architecture**:
  - Training both **EfficientNet-B0** and **ConvNeXt-Tiny** starting from pre-trained weights.
  - Custom fully connected classification heads to output probabilities for the four target classes.
- **Class Imbalance Handling**:
  - Computes class distributions and applies specific **class weights** directly to the CrossEntropyLoss function to ensure fair penalization for misclassifications of minority classes.
- **Advanced Evaluation & Metrics**:
  - Includes metrics such as **Accuracy, Precision, Recall, and F1-Score**.
  - Visual validations: 
    - Confusion Matrix (absolute and normalized)
    - ROC Curves & AUC (One-vs-Rest)
    - Multi-class Reliability Diagrams and Calibration curves
    - Probability Confidence Distribution and Prediction Entropy Analysis.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd Covid19_Detection
   ```

2. **Install dependencies:**
   Make sure you have a suitable environment with `Python 3.8+` and a CUDA-capable machine. Install the required libraries:
   ```bash
   pip install torch torchvision timm matplotlib seaborn scikit-learn tqdm numpy pillow
   ```

3. **Data Preparation:**
   Download the dataset and extract it. 
   *(Note: The base path in the notebook expects the dataset to be in a Kaggle-like structure. Edit the `base_path` variable in the notebook directly if you're loading from a different local directory.)*

4. **Running the Pipeline:**
   Launch Jupyter Notebook or Jupyter Lab and open `COVID_19_Detection.ipynb`:
   ```bash
   jupyter notebook COVID_19_Detection.ipynb
   ```
   Execute the cells sequentially to begin preprocessing, training, and running model inference.

## Dataset Split Policy
The standard implementation uses `random_split` to divide the total Dataset into three parts:
- **Training Set**: 70%
- **Validation Set**: 15%
- **Test Set**: 15%

## Final Output
A `.pth` file is generated saving the best model weights. The notebook outputs graphical evaluations of simple vs. hard predictions, letting you observe where the model was correct/incorrect based on confidence thresholds.

## License
MIT License
