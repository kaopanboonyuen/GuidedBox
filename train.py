'''
# Author: Teerapong Panboonyuen (Kao Panboonyuen)
# Project: GuidedBox - Guided Box-based Pseudo Mask Learning for Instance Segmentation
# Description: Training script for the GuidedBox model, implementing the proposed teacher-student architecture.
# License: MIT License
'''

import torch
from torch.utils.data import DataLoader
from model import GuidedBoxModel
from dataset import CustomDataset
import yaml
import os

# Load configuration
config_path = 'guidedbox_config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Initialize dataset and dataloader
# Example: Using Massachusetts Roads Dataset
# Ensure the dataset path and any necessary preprocessing transforms are defined in the config file
# Massachusetts Roads Dataset is used for segmentation tasks, providing satellite images and corresponding road masks
train_dataset = CustomDataset(config['train_data_path'], transform=config['train_transforms'])
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

# Initialize model
# GuidedBoxModel implements a teacher-student architecture for pseudo mask learning
model = GuidedBoxModel(config)
model = model.to(config['device'])

# Set optimizer
# Adam optimizer is chosen here for its adaptive learning rate capabilities
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

# Training loop
# Iterate over the number of epochs defined in the configuration
for epoch in range(config['num_epochs']):
    model.train()  # Set model to training mode
    for images, boxes, masks in train_loader:
        images = images.to(config['device'])  # Move images to the configured device (CPU/GPU)
        boxes = boxes.to(config['device'])    # Move ground truth bounding boxes to the device
        masks = masks.to(config['device'])    # Move ground truth masks to the device
        
        # Forward pass through the model
        loss = model(images, boxes, masks)  # Model returns a loss value during training
        
        # Backward pass
        optimizer.zero_grad()  # Clear the gradients
        loss.backward()        # Compute gradients based on the loss
        optimizer.step()       # Update model parameters
        
    # Log the loss for each epoch
    print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {loss.item()}')
    
# Save the trained model state to the specified path
torch.save(model.state_dict(), os.path.join(config['save_path'], 'guidedbox_model.pth'))