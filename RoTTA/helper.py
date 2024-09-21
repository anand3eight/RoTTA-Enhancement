import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def get_confmat_values(confmat):
    TP = np.diag(confmat)
    FP = np.sum(confmat, axis=0) - TP
    FN = np.sum(confmat, axis=1) - TP
    TN = np.sum(confmat) - (FP + FN + TP)

    return TP, FP, FN, TN

def save_confmat_and_metrics(y_true, y_pred, model_name, attack_type, loss, output_folder):
    # Calculate confusion matrix
    confmat = confusion_matrix(y_true, y_pred)

    formatted_confmat = "{\n" + "\n".join(
        "    {" + ", ".join(map(str, row)) + "}" for row in confmat
        ) + "\n}"
    
    # Save confusion matrix to file
    confmat_file = f"../{output_folder}/{model_name}_{attack_type}_confmat.txt"
    with open(confmat_file, 'w') as f:
        f.write(formatted_confmat)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')  # This is sensitivity (TPR)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # False positive rate (FPR) = FP / (FP + TN)
    # False negative rate (FNR) = FN / (FN + TP)
    tn, fp, fn, tp = get_confmat_values(confmat)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    # Store metrics in a dictionary
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,  # TPR
        'f1_score': f1,
        'fpr': fpr,
        'fnr': fnr,
        'loss' : loss
    }
    
    # Save metrics dictionary to file
    metrics_file = f"../{output_folder}/{model_name}_{attack_type}_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write(str(metrics_dict))
    
    print(f"Confusion matrix and metrics saved for model: {model_name}")

def evaluate_tta(loader, tta_model, model_name, attack_type):
    output_folder = 'Metrics'
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_preds = []
    total_loss = 0.0

    tbar = tqdm(loader)
    
    for batch_id, (images, labels) in enumerate(tbar):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images, labels = images.to(device), labels.to(device)
                
        output = tta_model(images)
        
        # Predictions and loss calculation
        loss = F.cross_entropy(output, labels)
        total_loss += loss.item()

        predict = torch.argmax(output, dim=1)
        accurate = (predict == labels).sum().item()
        correct_predictions += accurate
        total_predictions += labels.size(0)

        # Collecting labels and predictions for metrics calculation
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predict.cpu().numpy())
        
        # Update accuracy in the progress bar
        current_accuracy = correct_predictions / total_predictions
        tbar.set_postfix(accuracy=current_accuracy)

    # Calculate final accuracy
    final_accuracy = correct_predictions / total_predictions
    
    # Calculate average loss
    avg_loss = total_loss / total_predictions
    print(f"Final Accuracy : {final_accuracy}")
    print(f"Average Loss: {avg_loss}")

    # Call the confmat_and_metrics function
    save_confmat_and_metrics(all_labels, all_preds, model_name, attack_type, avg_loss, output_folder)

    return final_accuracy, avg_loss