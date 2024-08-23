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
train_dataset = CustomDataset(config['train_data_path'], transform=config['train_transforms'])
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])

# Initialize model
model = GuidedBoxModel(config)
model = model.to(config['device'])

# Set optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

# Training loop
for epoch in range(config['num_epochs']):
    model.train()
    for images, boxes, masks in train_loader:
        images = images.to(config['device'])
        boxes = boxes.to(config['device'])
        masks = masks.to(config['device'])
        
        # Forward pass
        loss = model(images, boxes, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{config["num_epochs"]}], Loss: {loss.item()}')
    
torch.save(model.state_dict(), os.path.join(config['save_path'], 'guidedbox_model.pth'))