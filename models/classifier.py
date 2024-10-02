import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score, f1_score


def eval_classifier(model, data_loader, config, device, use_text):
    """
    Evaluate the classifier model on the given data loader.

    Args:
        model (nn.Module): The classifier model to evaluate.
        data_loader (DataLoader): The data loader containing the evaluation data.
        config (dict): Configuration dictionary containing model parameters.
        device (torch.device): The device to run the evaluation on.
        use_text (bool): Whether to use text data alongside images.

    Returns:
        tuple: A tuple containing:
            - avg_loss (float): The average loss over the evaluation data.
            - metric (float): The evaluation metric (accuracy for single-label, F1-score for multi-label).
    """
    model.eval()
    task = config["task"]
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            if use_text:
                text = batch['text_description'].to(device)
                outputs = model(images, text)
            else:
                outputs = model(images)
            labels = batch['single_label'].to(device) if task=="single_label_classification" else batch["multi_label"].to(device)
            if task=="single_label_classification":
                predicted = outputs.argmax(dim=-1)
            elif task=="multi_label_classification":
                predicted = torch.sigmoid(outputs) > 0.5
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss/len(data_loader)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    if task=="single_label_classification":
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, acc
    if task=="multi_label_classification":
        f1_score = f1_score(all_labels, all_preds, average='samples')
        return avg_loss, f1_score
    
    
def train_classifier(model, train_loader, val_loader, config, device, use_text):
    """
    Train the classifier model using the provided training and validation data.

    Args:
        model (nn.Module): The classifier model to train.
        train_loader (DataLoader): The data loader containing the training data.
        val_loader (DataLoader): The data loader containing the validation data.
        config (dict): Configuration dictionary containing training parameters.
        device (torch.device): The device to run the training on.
        use_text (bool): Whether to use text data alongside images.

    Returns:
        nn.Module: The trained model.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['classification']['model_training']["learning_rate"])
    task = config["task"]
    use_text = config["use_text"]
    
    if task == "single_label_classification":
        criterion = nn.CrossEntropyLoss()
    if task == "multi_label_classification":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = None
    

    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0
        val_loss = 0
        train_preds = []
        train_labels = []
        for batch in train_loader:
            optimizer.zero_grad()
            if task == 'entity_tagging':
                #Entity tagging logic
                pass
            images = batch['image'].to(device)
            if use_text:
                text = batch['text_description'].to(device)
                outputs = model(images, text)
            else:
                outputs = model(images)
            
            labels = batch['single_label'].to(device) if task=="single_label_classification" else batch["multi_label"].to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if task=="single_label_classification":
                _, predicted = torch.max(outputs.data, 1)
        
            elif task=="multi_label_classification":
                predicted = torch.sigmoid(outputs) > 0.5
                
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        epoch_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {epoch_train_loss:.4f}")
        #Validation loss
        val_metric, val_loss = eval_classifier(model, val_loader, config, device, use_text)
        if task=="single_label_classification":
            acc = accuracy_score(train_labels, train_preds)
        elif task=="multi_label_classification":
            f1_score = f1_score(train_labels, train_preds, average='samples')
        
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            torch.save(model.state_dict(), config["model_path"])  
    return model


def inference_classifier(model, data_loader, config, device, use_text):
    """
    Perform inference using the trained classifier model.

    Args:
        model (nn.Module): The trained classifier model.
        data_loader (DataLoader): The data loader containing the inference data.
        config (dict): Configuration dictionary containing model parameters.
        device (torch.device): The device to run the inference on.
        use_text (bool): Whether to use text data alongside images.

    Returns:
        Not specified in the given code (function is a placeholder).
    """
    pass