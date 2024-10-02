import difflib
import torch
import torch
from utils.model_utils import EntityTagger

def map_nutrients(extracted_text, nutrient_labels):
    """
    Map extracted text to nutrient labels using fuzzy string matching.

    Args:
        extracted_text (str): The text extracted from an image or document.
        nutrient_labels (list): A list of known nutrient labels to match against.

    Returns:
        dict: A dictionary where keys are nutrient labels and values are 1 (present) or 0 (absent).

    Note:
        This function uses a cutoff of 0.7 for string matching, which may need to be adjusted
        based on the specific requirements of your application.
    """
    mapped_nutrients = {}
    for nutrient in nutrient_labels.lower():
        matches = difflib.get_close_matches(nutrient, nutrient_labels, n=1, cutoff=0.7)
        if matches:
            mapped_nutrients[nutrient] = 1
        else:
            mapped_nutrients[nutrient] = 0
    
    return mapped_nutrients

def get_nutrients(mapped_nutrients):
    """
    Convert the mapped nutrients dictionary to a structured data format.

    Args:
        mapped_nutrients (dict): A dictionary of mapped nutrients where keys are nutrient names
                                 and values are 1 (present) or 0 (absent).

    Returns:
        dict: A dictionary with two keys:
              - "Nutrient Entity": A list of nutrient names.
              - "Presence": A list of corresponding presence values (1 or 0).
    """
    data = {"Nutrient Entity": list(mapped_nutrients.keys()), "Presence": list(mapped_nutrients.values())}
    return data


def get_ner_output(dataloader, device):
    """
    Perform Named Entity Recognition (NER) on a batch of images using the EntityTagger.

    Args:
        dataloader (torch.utils.data.DataLoader): A DataLoader containing batches of images.
        device (torch.device): The device (CPU or GPU) to run the model on.

    Returns:
        list: A list of predictions for each batch in the dataloader. The structure of these
              predictions depends on the implementation of EntityTagger.predict().

    Note:
        This function assumes that the EntityTagger class has a static 'predict' method
        that can process batches of images.
    """
    all_prediction = []
    for batch in dataloader:
        predictions = EntityTagger.predict(batch['image'].to(device))
        all_prediction.append(predictions)
    return all_prediction
        
