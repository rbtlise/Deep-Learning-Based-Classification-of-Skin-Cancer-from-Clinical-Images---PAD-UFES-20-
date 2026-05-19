# Deep-Learning-Based-Classification-of-Skin-Cancer-from-Clinical-Images---PAD-UFES-20-

Here is a clean, professional, and well-structured README.md file tailored specifically for your multimodal skin cancer classification project.

# Multimodal Skin Cancer Classification Platform

An advanced deep learning framework designed to classify skin lesions into six distinct pathologies using a parallel multimodal architecture. The platform combines deep spatial features from clinical smartphone images with structured patient metadata (demographics, clinical history) to optimize diagnostic sensitivity.

## Architecture Overview

The system processes data through two parallel extraction pipelines before performing late-stage fusion:
1. **Visual Branch:** Utilizes a pre-trained ResNet-50 backbone (with classification head replaced by `nn.Identity()`) to extract a 2048-dimensional spatial feature vector.
2. **Clinical Tabular Branch:** Implements a Multi-Layer Perceptron (MLP) with Dense layers and Dropout to extract a 32-dimensional metadata feature embedding.

The vectors are concatenated into a unified 2080-dimensional representation before being evaluated by a fully connected classification head mapping across six skin lesion categories.

---

## Prerequisites

Ensure you have Python 3.8+ and a CUDA-capable GPU environment (highly recommended for training acceleration). 

Install the required dependencies:
```bash
pip install torch torchvision pandas numpy scikit-learn pillow matplotlib seaborn
File Structure
Your workspace folder structure should match the following configuration:

Plaintext
├── metadata.csv                 # Raw clinical metadata file
├── images/                      # Directory containing clinical smartphone photos (.jpg)
├── Skin_Cancer_main.py          # Main execution script
├── Skin_Cancer_EDA.py           # Exploratory Data ANalysis execution script
└── README.md                    # Project documentation
Configuration Setup
Before executing the pipeline, check the CONFIG dictionary and file paths inside main.py.

Dataset Location: Ensure the global path points directly to your Google Drive or local drive environment:

Python
path = "/content/drive/MyDrive/ML_project/metadata.csv"
Device Selection: The script maps optimizations to CUDA if available, defaulting otherwise to CPU:

Python
CONFIG = {
    'image_size': 224,
    'batch_size': 32,
    'lr': 0.0001,
    'n_splits': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
How to Launch
Follow these steps to run data processing, cross-validation training, and evaluation phases:

1. Prepare and Authenticate (If using Google Colab)
If running inside a Jupyter or Google Colab notebook, mount your Google Drive workspace first:

Python
from google.colab import drive
drive.mount('/content/drive')

2. Execute the Pipeline
Run the main script from your system terminal or execution cell:

Bash
python main.py

3. What Happens During Execution
Upon launching, the script will sequentially execute the following processing stages:

Data Cleansing & Preprocessing: Drops duplicate rows, dropped administrative identifiers (patient_id), and applies mean/mode missing value imputation.

Stratified K-Fold Generation: Splits data into 5 distinct folds while keeping global class balance constant across training/validation sets.

Leakage-Free Scaling: Fits a StandardScaler on the training partition age data and applies it to the hidden validation partition.

Model Training: Optimizes network weights over 20 epochs using the Adam optimizer and enforces Weighted Cross-Entropy Loss to penalize minority-class mistakes.

Evaluation Output: Automatically calculates and displays per-class Sensitivity (Recall), Specificity, and prints out a comprehensive Confusion Matrix for each active fold.


### Direct Customization Tips:
* **Images Directory:** If your dataset loader expects a custom mapping folder or a diction
