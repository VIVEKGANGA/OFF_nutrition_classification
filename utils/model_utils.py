from PIL import Image
import boto3
import cv2
import torch.nn as nn
from transformers import BertTokenizer, BertForTokenClassification, BertModel
import torchvision
import torch
from sklearn.metrics import accuracy_score, f1_score

class OCREngine:
    """
    A class to perform OCR using Amazon Textract """
    def __init__(self):
        self.textract_client = boto3.client('textract')
    
    def extract_text_from_image(self, image):
        """
        Extracts text from an image using Textract.
        :param image_path: Path to the image file.
        :return: Extracted text from the image.
        """
        # Open the image
        image  = self.preprocess_image(image)
        # Extract text from the image using Textract
        extracted_text = "Nutritional Information from text"
        parsed_output = self._parse_response(extracted_text)
        return parsed_output
    
    def preprocess_image(self, image):
        """
        Preprocess the input image for better OCR results.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        return processed_image
    
    def _parse_response(self, response):
        """
        Parses the Textract response to extract text.
        :param response: Textract response.
        :return: Extracted text.
        """
        # Implement your parsing logic here
        parsed_text = "Parsed Text"
        return parsed_text


class EntityTagger(nn.Module):
    def __init__(self, config):
        """
        Initialize the EntityTagger.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super(EntityTagger, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        num_classes = config['entity_tagging']['num_labels']
        self.model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
        self.model.eval()
        
        self.label = config['Multimodal']['label']
    
    def predict(self, image):
        """
        Predict entities in the given image.

        Args:
            image: The input image.

        Returns:
            list: Predicted entities.
        """
        ocr_engine = OCREngine()
        self.extracted_text = ocr_engine.extract_text_from_image(image)
        inputs = self.tokenizer(self.extracted_text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
        predictions = predictions.cpu().numpy()
        input_ids = inputs['input_ids'].cpu().numpy()
        results = []
        for pred, input_id in zip(predictions, input_ids):
            if self.tokenizer.convert_ids_to_tokens(input_id) in self.label:
                results.append(self.label[pred])
        return results
    
class Textencoder(nn.Module):
    """
    A neural network module for encoding text using BERT.
    """
    def __init__(self):
        """
        Initialize the Textencoder with a pre-trained BERT model.
        """
        super(Textencoder, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the Textencoder.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Encoded text features.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output
    
class ImageEncode(nn.Module):
    """
    A neural network module for encoding images using ResNet18.
    """
    def __init__(self):
        """
        Initialize the ImageEncode with a pre-trained ResNet18 model.
        """
        super(ImageEncode, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()
    
    def forward(self, x):
        """
        Forward pass of the ImageEncode.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Encoded image features.
        """
        return self.resnet(x)
        
        
class SingleLabelImageClassifier(nn.Module):
    """
    A neural network module for single-label image classification.
    """

    def __init__(self, num_classes):
        """
        Initialize the SingleLabelImageClassifier.

        Args:
            num_classes (int): Number of classes for classification.
        """
        super(SingleLabelImageClassifier, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the SingleLabelImageClassifier.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Classification logits.
        """
        x = self.resnet(x)
        x = self.fc(x)
        return x
    
class MultiLabelImageClassifier(nn.Module):
    """
    A neural network module for multi-label image classification.
    """
    def __init__(self, num_classes):
        """
        Initialize the MultiLabelImageClassifier.

        Args:
            num_classes (int): Number of classes for classification.
        """
        super(MultiLabelImageClassifier, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the MultiLabelImageClassifier.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Classification probabilities.
        """
        x = self.resnet(x)
        x = self.fc(x)
        return torch.sigmoid(x)
    
class MultimodalSingleLabelClassifier(nn.Module):
    """
    A neural network module for single-label classification using both image and text data.
    """

    def __init__(self, num_classes):
        """
        Initialize the MultimodalSingleLabelClassifier.

        Args:
            num_classes (int): Number of classes for classification.
        """
        super(MultimodalSingleLabelClassifier, self).__init__()
        self.text_encoder = Textencoder()
        self.image_encoder = ImageEncode()
        self.fc = nn.Linear(self.text_encoder.model.config.hidden_size + self.image_encoder.resnet.fc.in_features, num_classes)

    def forward(self, image, text):
        """
        Forward pass of the MultimodalSingleLabelClassifier.

        Args:
            image (torch.Tensor): Input image tensor.
            text (dict): Input text data.

        Returns:
            torch.Tensor: Classification logits.
        """
        text_features = self.text_encoder(**text)
        image_features = self.image_encoder(image)
        combined_features = torch.cat([text_features, image_features], dim=1)
        output = self.fc(combined_features)
        return output


class MultimodalMultiLabelClassifier(nn.Module):
    """
    A neural network module for multi-label classification using both image and text data.
    """
    def __init__(self, num_classes):
        """
        Initialize the MultimodalMultiLabelClassifier.

        Args:
            num_classes (int): Number of classes for classification.
        """
        super(MultimodalMultiLabelClassifier, self).__init__()
        self.text_encoder = Textencoder()
        self.image_encoder = ImageEncode()
        self.fc = nn.Linear(self.text_encoder.model.config.hidden_size + self.image_encoder.resnet.fc.in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image, text):
        """
        Forward pass of the MultimodalMultiLabelClassifier.

        Args:
            image (torch.Tensor): Input image tensor.
            text (dict): Input text data.

        Returns:
            torch.Tensor: Classification probabilities.
        """
        text_features = self.text_encoder(**text)
        image_features = self.image_encoder(image)
        combined_features = torch.cat([text_features, image_features], dim=1)
        output = self.fc(combined_features)
        output = self.sigmoid(output)
        return output