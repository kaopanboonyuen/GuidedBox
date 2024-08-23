'''
# Author: Teerapong Panboonyuen (Kao Panboonyuen)
# Project: GuidedBox - Guided Box-based Pseudo Mask Learning for Instance Segmentation
# Description: Inference script for running the GuidedBox model on new data.
# License: MIT License
'''

import torch
from model import GuidedBoxModel
from PIL import Image
import yaml

# Load configuration
config_path = 'guidedbox_config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Load model
model = GuidedBoxModel(config)
model.load_state_dict(torch.load(config['model_checkpoint']))
model = model.to(config['device'])
model.eval()

# Inference function
def run_inference(image_path):
    image = Image.open(image_path).convert('RGB')
    image = config['eval_transforms'](image).unsqueeze(0).to(config['device'])
    
    with torch.no_grad():
        predicted_masks = model(image)
    
    return predicted_masks

# Example usage
image_path = 'path/to/your/image.jpg'
predicted_masks = run_inference(image_path)
# Save or display the predicted masks as needed