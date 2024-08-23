'''
# Author: Teerapong Panboonyuen (Kao Panboonyuen)
# Project: GuidedBox - Guided Box-based Pseudo Mask Learning for Instance Segmentation
# Description: Evaluation script for the GuidedBox model using various metrics.
# License: MIT License
'''

import torch
from torch.utils.data import DataLoader
from model import GuidedBoxModel
from dataset import CustomDataset
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def compute_iou(preds, masks):
    """
    Computes the Intersection over Union (IoU) between predicted masks and ground truth masks.

    Parameters:
    - preds: Predicted masks (tensor)
    - masks: Ground truth masks (tensor)

    Returns:
    - iou: Mean IoU value
    """
    intersection = (preds & masks).float().sum((1, 2))
    union = (preds | masks).float().sum((1, 2))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

# Load configuration
config_path = 'guidedbox_config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Initialize dataset and dataloader
eval_dataset = CustomDataset(config['eval_data_path'], transform=config['eval_transforms'])
eval_loader = DataLoader(eval_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

# Initialize model
model = GuidedBoxModel(config)
model.load_state_dict(torch.load(config['model_checkpoint'], map_location=config['device']))
model = model.to(config['device'])
model.eval()

# Evaluation loop
total_iou = 0
total_precision = 0
total_recall = 0
total_f1 = 0
total_accuracy = 0
count = 0

with torch.no_grad():
    for images, boxes, masks in eval_loader:
        images = images.to(config['device'])
        masks = masks.to(config['device'])
        
        # Forward pass
        predicted_masks = model(images)
        predicted_masks = torch.sigmoid(predicted_masks) > 0.5  # Binarize the predictions
        
        # Compute IoU
        iou = compute_iou(predicted_masks, masks)
        total_iou += iou.item()
        
        # Compute precision, recall, F1 score, accuracy
        y_true = masks.cpu().numpy().flatten()
        y_pred = predicted_masks.cpu().numpy().flatten()
        total_precision += precision_score(y_true, y_pred)
        total_recall += recall_score(y_true, y_pred)
        total_f1 += f1_score(y_true, y_pred)
        total_accuracy += accuracy_score(y_true, y_pred)
        
        count += 1

    print(f'Mean IoU: {total_iou / count:.4f}')
    print(f'Mean Precision: {total_precision / count:.4f}')
    print(f'Mean Recall: {total_recall / count:.4f}')
    print(f'Mean F1 Score: {total_f1 / count:.4f}')
    print(f'Mean Accuracy: {total_accuracy / count:.4f}')