# Deep-Learning-Based-Classification-of-Skin-Cancer-from-Clinical-Images---PAD-UFES-20-

# Multimodal Skin Cancer Classification Platform

An advanced deep learning framework designed to classify skin lesions into six distinct pathologies using a parallel multimodal architecture. The platform combines deep spatial features from clinical smartphone images with structured patient metadata (demographics, clinical history) to optimize diagnostic sensitivity.

## Architecture Overview

The system processes data through two parallel extraction pipelines before performing late-stage fusion:
1. **Visual Branch:** Utilizes a pre-trained ResNet-50 backbone (with classification head replaced by `nn.Identity()`) to extract a 2048-dimensional spatial feature vector.
2. **Clinical Tabular Branch:** Implements a Multi-Layer Perceptron (MLP) with Dense layers and Dropout to extract a 32-dimensional metadata feature embedding.

The vectors are concatenated into a unified 2080-dimensional representation before being evaluated by a fully connected classification head mapping across six skin lesion categories.


## Prerequisites

Ensure you have Python 3.8+ and a CUDA-capable GPU environment (highly recommended for training acceleration). 

Install the required dependencies:
```bash
pip install torch torchvision pandas numpy scikit-learn pillow matplotlib seaborn
```
File Structure
Your workspace folder structure should match the following configuration:

```bash
├── metadata.csv                 # Raw clinical metadata file
├── images/                      # Directory containing clinical smartphone photos (.jpg)
├── Skin_Cancer_main.py          # Main execution script
├── Skin_Cancer_EDA.py           # Exploratory Data ANalysis execution script
└── README.md                    # Project documentation
```

You can also store metadat.csv and images in other folders.

**Configuration Setup**
Before running the script : Prepare and link dataset in Skin_Cancer_main (matadata and images) and Skin_Cancer_EDA (metadata)

If running inside a Jupyter or Google Colab notebook, mount your Google Drive workspace first:

```bash
from google.colab import drive
drive.mount('/content/drive')
```

Add the path to your images in the main script (image path)

Add the path to your metadata in the main script (metadata path)
