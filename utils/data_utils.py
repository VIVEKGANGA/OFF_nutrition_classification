import torch
import cv2
import os
from io import BytesIO
import pandas as pd
from PIL import Image
import requests 
from torch.utils.data import Dataset, DataLoader


class NutritionDataset(Dataset):
    """
    A custom Dataset class for handling nutritional data.

    This class prepares and manages data for nutritional image classification tasks,
    including downloading images and handling text descriptions if required.

    Attributes:
        config (dict): Configuration dictionary containing dataset parameters.
        product_data (pd.DataFrame): DataFrame containing product information.
        image_dir (str): Directory to store downloaded images.
        data (pd.DataFrame): Processed data ready for use.
        task (str): The type of classification task.
        nutritional_image_column (str): Column name for nutritional image URLs.
        use_text (bool): Whether to use text data alongside images.
        text_columns (str): Column name for text descriptions (if use_text is True).
    """
    def __init__(self, product_data, config):
        self.config = config
        self.product_data = product_data
        self.image_dir = config['dataset']['image_dir']
        self.data = self.prepare_data()
        self.task = config['task']
        self.nutritional_image_column = self.config['dataset']['nutitional_image_columns']
        self.use_text = self.config['Multimodal']['use_text']
        if self.use_text:
            self.text_columns = self.config['Multimodal']['text_column']
    
    
    def prepare_data(self):
        """
        Prepare the dataset by downloading images and organizing data.

        Returns:
            pd.DataFrame: Processed data ready for use in the dataset.
        """
        nutrional_data = []
        for idx, row in self.product_data.iterrows():
            nutritional_image_url = row[self.nutritional_image_column]
            if pd.isna(nutritional_image_url):
                continue
            image_path = self.download_image(nutritional_image_url)
            if self.use_text:
                text_description = row[self.text_columns]
            data_dict = {
                'image_path': image_path,
                'product_name': row['product_name'],
                'text_description': text_description
            }
            nutrional_data.append(data_dict)
        return pd.DataFrame(nutrional_data)
        
    def download_image(self, image_url):
        """
        Download an image from a given URL and save it locally.

        Args:
            image_url (str): URL of the image to download.

        Returns:
            str: Local path of the downloaded image, or None if download failed.
        """
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            image_path = self.image_dir + image_url.split('/')[-1]
            image.save(image_path)
            return image_path
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        product = self.data.iloc[idx]
        image = Image.open(product['image_path']).convert('RGB')
        if self.use_text:
            text_description = product['text_description']
        else:
            text_data = None

        if self.task=='single_label_classification':
            return {"image": image, "label": product['single_label'],"text_description":text_description}
        elif self.task=='multi_label_classification':
            return {"image": image, "label": product['multi_label'],"text_description":text_description}
        elif self.task=='entity_tagging':
            return {"image": image, "annotation": product['annotation'],"text_description":None}
        

def get_DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the given dataset.

    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int, optional): Number of samples per batch. Defaults to 10.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.

    Returns:
        DataLoader: A DataLoader object for the given dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def split_data(dataset, method, params):
    """
    Split the dataset using the specified method and parameters.

    Args:
        dataset (Dataset): The dataset to split.
        method (str): The splitting method to use ('random', 'stratified', or 'custom').
        params (dict): Parameters for the splitting method.

    Returns:
        tuple: Split datasets.

    Raises:
        ValueError: If an invalid split method is provided.
    """
    if method=='random':
        return random_split(dataset, params)
    if method=='stratified':
        return stratifiedSplit(dataset, params)
    if method=='custom':
        return customSplit(dataset, params)
    else:
        raise ValueError("Invalid split method")

def random_split(dataset, params):
    """
    Perform a random split of the dataset.

    Args:
        dataset (Dataset): The dataset to split.
        params (dict): Parameters for the random split.

    Returns:
        tuple: Randomly split datasets.
    """
    pass

def stratifiedSplit(dataset, params):
    """
    Perform a stratified split of the dataset.

    Args:
        dataset (Dataset): The dataset to split.
        params (dict): Parameters for the stratified split.

    Returns:
        tuple: Stratified split datasets.
    """
    pass

def customSplit(dataset, params):
    """
    Perform a custom split of the dataset.

    Args:
        dataset (Dataset): The dataset to split.
        params (dict): Parameters for the custom split.

    Returns:
        tuple: Custom split datasets.
    """
    pass

