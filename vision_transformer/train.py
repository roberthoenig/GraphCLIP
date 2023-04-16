import wandb
wandb.login()
wandb.init(project='ResearchAssistant', entity='jthomm')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import sys
sys.path.append(".")
from jt_training import get_dataloader, train_one_epoch, evaluate, get_free_gpu
from open_clip.jt_ViT_RelClassifier import ViT_RelClassifier
from itertools import chain
import torch.optim as optim
import os
from datetime import datetime


############################################################################
image_dir = "/local/home/jthomm/GraphCLIP/datasets/visual_genome/raw/VG/"
metadata_path = "/local/home/jthomm/GraphCLIP/datasets/visual_genome/processed/"
run_logs_dir = "/local/home/jthomm/GraphCLIP/experiments/"

num_epochs = 10
clip_model_type = 'ViT-B/32'
clip_pretrained_dataset = 'laion400m_e32'
description = f"""
    ViT_RelClassifier with 100 epochs, 200 hidden size, and 64 batch size\n 
    clip model {clip_model_type}, clip pretrained dataset {clip_pretrained_dataset}
    the model has three heads: rel, obj_1, obj_2
    the training rates are: ViT: 1e-6, rest: 1e-4
    """
debug_mode = False # turn this on such that a tiny dataset is loaded such that you can test the code
############################################################################


### setup the logging and checkpointing

date_folder = run_logs_dir + datetime.now().strftime("%Y-%m-%d")
os.makedirs(date_folder, exist_ok=True)

run_id = len(os.listdir(date_folder))
run_name = f"vision_transformer_{run_id}"
run_folder = f"{date_folder}/{run_name}"
os.makedirs(run_folder)

wandb.init(project='ResearchAssistant', entity='jthomm', dir=run_folder, notes=description)

checkpoint_path = f"{run_folder}/checkpoints"
os.makedirs(checkpoint_path)

### setup the model

model = ViT_RelClassifier(100, 200, clip_model_type, clip_pretrained_dataset)
prepocess_function = model.preprocess
# Load your dataloader
dataloader_train, dataloader_val = get_dataloader(prepocess_function,metadata_path,image_dir, testing_only=debug_mode)

# Move the model to the GPU
device =get_free_gpu()
print(f"Using device {device}")
wandb.config.device = str(device)
model = model.to(device)

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model.ViT.parameters(), 'lr': 1e-6},
    {'params': chain(model.rel_classifier.parameters(), model.obj_1_classifier.parameters(), model.obj_2_classifier.parameters()), 'lr': 1e-4},
    {'params': chain(model.class_conv.parameters(), model.bounding_boxes_map.parameters()), 'lr': 1e-4},
])

# Set up the gradient scaler for mixed precision training
scaler = GradScaler()

# Training loop
best_val_acc = 0.0
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_one_epoch(model, dataloader_train, criterion, optimizer, device, scaler, epoch)
    val_rel_acc, val_obj_1_acc, val_obj_2_acc =  evaluate(model, dataloader_val, criterion, device, epoch)
    
    if val_rel_acc > best_val_acc:
        best_val_acc = val_rel_acc
        torch.save(model.state_dict(), f"{checkpoint_path}/best_rel_model.pt")

print("Training complete.")
