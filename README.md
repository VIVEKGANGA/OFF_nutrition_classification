# Nutrition Image Classification and Entity Tagging

This package provides a set of tools for nutrition image classification and entity tagging using deep learning techniques. It supports single-label and multi-label classification, as well as entity tagging for nutritional information extraction from images.

## Note

This package contains pseudo-code and may require additional implementation and refinement for full functionality.

## Features

- Single-label image classification
- Multi-label image classification
- Multimodal (image + text) classification
- Entity tagging for nutritional information
- Support for both training and inference modes
- Configurable via YAML file

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- transformers
- pandas
- numpy
- scikit-learn
- opencv-python
- Pillow
- boto3 (for AWS Textract integration)
- PyYAML

## Installation

    Clone this repository and install the required packages:

    ```bash
    git clone https://github.com/VIVEKGANGA/nutrition-classification.git
    cd nutrition-classification
    pip install -r requirements.txt
    ```


## Installation
1. Prepare your dataset and update the Config.yaml file with appropriate settings.

2. Run the main script:
    python main.py

## Configuration

The Config.yaml file contains all the necessary settings for the models and data processing. Key configurations include:

- Dataset paths

- Model selection (single-label, multi-label, entity tagging)

- Training parameters (batch size, learning rate, etc.)

- Multimodal settings (use of text data)

## Project Structure
 - main.py: Entry point of the application

- utils/: Utility functions for data processing and model creation

- models/: Implementation of classifier and entity tagger

- Config.yaml: Configuration file

## Model 

- SingleLabelImageClassifier: For single-label image classification

- MultiLabelImageClassifier: For multi-label image classification

- MultimodalSingleLabelClassifier: For single-label classification using both image and text data

- MultimodalMultiLabelClassifier: For multi-label classification using both image and text data

- EntityTagger: For entity tagging in nutritional images


## Data Processing
The NutritionDataset class handles data loading and preprocessing, including image downloading and text processing if required.

## Training and Inference
The package supports both training and inference modes, controlled by the is_training flag in the configuration file.

## Entity Tagging
For entity tagging tasks, the package uses AWS Textract for OCR and a BERT-based model for token classification.

## Limitations and Future Work
- This package contains pseudo-code and requires further implementation for full functionality.

- Error handling and edge cases may need additional attention.

- Performance optimization may be necessary for large-scale applications.

- Integration with a web interface or API could enhance usability.