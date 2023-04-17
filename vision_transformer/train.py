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
from jt_training import get_dataloader, train_one_epoch, evaluate, get_free_gpu, get_all_free_gpus_ids
from open_clip.jt_ViT_RelClassifier import ViT_RelClassifier
from open_clip.transform import image_transform
from itertools import chain
import torch.optim as optim
import os
from datetime import datetime


############################################################################
image_dir = "/local/home/jthomm/GraphCLIP/datasets/visual_genome/raw/VG/"
metadata_path = "/local/home/jthomm/GraphCLIP/datasets/visual_genome/processed/"
run_logs_dir = "/local/home/jthomm/GraphCLIP/experiments/"

num_epochs = 10
clip_model_type = 'ViT-H-14'
clip_pretrained_dataset = 'laion2b_s32b_b79k'
description = f"""
    ViT_RelClassifier with 100 epochs, 200 hidden size, and 64 batch size\n 
    clip model {clip_model_type}, clip pretrained dataset {clip_pretrained_dataset}
    the model has three heads: rel, obj_1, obj_2
    the training rates are: ViT: 1e-6, rest: 1e-4
    """
debug_mode = True # turn this on such that a tiny dataset is loaded such that you can test the code
############################################################################

### setup CUDA
free_devices = get_all_free_gpus_ids()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(i) for i in free_devices])
print(f"Making CUDA devices visible: {free_devices}")

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

model = nn.DataParallel(ViT_RelClassifier(100, 200, clip_model_type, clip_pretrained_dataset))
image_size = model.module.visual.image_size
print(f"Image size: {image_size}")
preprocess_function = image_transform(
    image_size,
    is_train=True,
    mean=None,
    std=None,
    aug_cfg=None,
)

# preprocess_function = model.module.preprocess
# Load your dataloader
dataloader_train, dataloader_val = get_dataloader(preprocess_function,metadata_path,image_dir,batch_size=64, testing_only=debug_mode)

# Move the model to the GPU
# device =get_free_gpu()
# print(f"Using device {device}")
# wandb.config.device = str(device)
# model = model.to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.cuda()
wandb.config.num_gpus = torch.cuda.device_count()

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model.module.ViT.parameters(), 'lr': 1e-6},
    {'params': chain(model.module.rel_classifier.parameters(), model.module.obj_1_classifier.parameters(), model.module.obj_2_classifier.parameters()), 'lr': 1e-4},
    {'params': chain(model.module.class_conv.parameters(), model.module.bounding_boxes_map.parameters()), 'lr': 1e-4},
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
        print(f"Saving model with best val acc: {best_val_acc} at epoch {epoch+1}")
        torch.save(model.state_dict(), f"{checkpoint_path}/best_rel_model_{epoch+1}.pt")

print("Training complete.")
