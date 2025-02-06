import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import torchattacks
from .adaptive_attacks import mixed_batch_attack, continuous_transformation_attack

def get_confmat_values(confmat):
    TP = np.diag(confmat)
    FP = np.sum(confmat, axis=0) - TP
    FN = np.sum(confmat, axis=1) - TP
    TN = np.sum(confmat) - (FP + FN + TP)

    return TP, FP, FN, TN

def save_confmat_and_metrics(y_true, y_pred, model_name, attack_type, loss):
    # Calculate confusion matrix
    confmat = confusion_matrix(y_true, y_pred)

    formatted_confmat = "{\n" + "\n".join(
        "    {" + ", ".join(map(str, row)) + "}" for row in confmat
        ) + "\n}"

    base_path = "/home/project/Documents/RoTTA-Enhancement/Sem8/"
    # Save confusion matrix to file
    confmat_file = f"{base_path}/Metrics/{model_name}_{attack_type}_confmat.txt"
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
    metrics_file = f"{base_path}/Metrics/{model_name}_{attack_type}_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write(str(metrics_dict))

    print(f"Confusion matrix and metrics saved for model: {model_name}")

def get_adversary(model, attack_type=None) :

      if (attack_type == "FGSM"):
          adversary = torchattacks.FGSM(model, eps=8/255)

      elif (attack_type == "PGD"):
          adversary = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)

      elif (attack_type == "CW"):
          adversary = torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)

      elif (attack_type == "APGD") :
          adversary = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, eot_iter=1, rho=.75, verbose=False)

      elif (attack_type == "APGD-t") :
          adversary = torchattacks.APGDT(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, eot_iter=1, rho=.75, verbose=False, n_classes=10)

      elif (attack_type == "FAB-t") :
          adversary = torchattacks.FAB(model, norm='Linf', steps=10, eps=8/255, n_restarts=1, alpha_max=0.1, eta=1.05, beta=0.9, verbose=False, seed=0, n_classes=10)

      elif (attack_type == "Square") :
          adversary = torchattacks.Square(model, norm='Linf', eps=8/255, n_queries=5000, n_restarts=1, p_init=.8, seed=0, verbose=False, loss='margin', resc_schedule=True)

      elif (attack_type == "AA") :
          adversary = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)

      elif (attack_type == "JSMA") :
          adversary = torchattacks.JSMA(model, theta=1.0, gamma=0.1)

      elif (attack_type == "EOTA") :
          adversary = torchattacks.EOTPGD(model, eps=8/255, alpha=2/255, steps=10, eot_iter=2)

      else:
          adversary = None

      return adversary

def evaluate_tta(loader, tta_model, model, model_name, attack_type):
    correct_predictions = 0
    total_predictions = 0
    all_labels = []
    all_preds = []
    total_loss = 0.0
    tbar = tqdm(loader)

    if attack_type != 'Clean' :
        adversary = get_adversary(model, attack_type)

    for _, (images, labels) in enumerate(tbar):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images, labels = images.to(device), labels.to(device)

        if (attack_type == "CTA") :
          images = continuous_transformation_attack(model, images, labels)

        elif (attack_type == "MBA") :
          images = mixed_batch_attack(model, images, labels)

        elif attack_type != 'Clean' :
          images = adversary(images, labels)

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
    save_confmat_and_metrics(all_labels, all_preds, model_name, attack_type, avg_loss)

    return final_accuracy, avg_loss