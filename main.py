import yaml
import logging
import pandas as pd
import torch
from utils.data_utils import NutritionDataset, split_dataset
from utils.model_utils import SingleLabelImageClassifier, MultiLabelImageClassifier, EntityTagger, MultimodalSingleLabelClassifier, MultimodalMultiLabelClassifier
from models.classifier import train_classifier, inference_classifier
from models.entity_tagger import get_ner_output
import numpy as np
from torch.utils.data import DataLoader
import os

def main(config_path='Config.yaml'):
    """
    Main function to run the nutrition classification or entity tagging task.

    This function loads the configuration, sets up the environment, prepares the data,
    initializes the appropriate model, and either trains the model or performs inference
    based on the configuration.

    Args:
        config_path (str): Path to the configuration YAML file. Defaults to 'Config.yaml'.

    Returns:
        None

    Raises:
        FileNotFoundError: If the config file or model artifacts are not found.
        ValueError: If an invalid task name is provided in the config.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(config['dataset']['data_dir'], config['dataset']['data_name'])
    data = pd.read_csv(data_path)
    task_name = config['task']
    if task_name == "entity_tagging":
        num_labels = config['entity_tagging']['num_labels']
    elif (task_name == "single_label_classification") | (task_name == "multi_label_classification"):
        num_classes = config['classification']['num_classes']
    else:
        logging.error("Invalid task name")
    dataset = NutritionDataset(data, config)
    use_text = config['Multimodal']['use_text']
    if (task_name == "single_label_classification") & (not use_text):
        model = SingleLabelImageClassifier(num_classes)
    elif (task_name == "multi_label_classification") & (not use_text) :
        model = MultiLabelImageClassifier(num_classes)
    elif (task_name == "single_label_classification") & (use_text) :
        model = MultimodalSingleLabelClassifier(num_classes)
    elif (task_name == "multi_label_classification") & (use_text) :
        model = MultimodalMultiLabelClassifier(num_classes)
    elif task_name == "entity_tagging":
        model = EntityTagger(config)
    model.to(device)

    if (task_name!='entity_tagging') & (config['is_training']):
        split_params = config['split_params']
        train_dataset, val_dataset = split_dataset(dataset, split_params)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)    
        train_classifier(model, train_loader, val_loader, config, device, use_text)
        
    elif (task_name!='entity_tagging') & (not config['is_training']):
        if os.path.exists(config['model_path']):
            model.load_state_dict(torch.load(config['model_path']))
        else: 
            logging.error("Model artifacts not found!")
        model_output = inference_classifier(model, val_loader, config, device)
        model_output.to_csv(config['dataset']['output_dir'])
    elif (task_name=='entity_tagging'):
        data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
        get_ner_output(data_loader, device)
    
if __name__ == "__main__":
    main("Config.yaml")
