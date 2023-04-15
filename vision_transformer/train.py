import wandb
wandb.login()
wandb.init(project='ResearchAssistant2', entity='jthomm')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import sys
sys.path.append(".")
from jt_training import get_dataloader, train_one_epoch
from open_clip.jt_ViT_RelClassifier import ViT_RelClassifier
from itertools import chain
import torch.optim as optim


############################################################################
image_dir = "/local/home/stuff/visual-genome/VG/"
metadata_path = "/local/home/jthomm/GraphCLIP/datasets/visual_genome/processed/"
############################################################################

model = ViT_RelClassifier(100, 200)
prepocess_function = model.preprocess
# Load your dataloader
dataloader = get_dataloader(prepocess_function,metadata_path,image_dir, testing_only=False)

# Move the model to the GPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model.ViT.parameters(), 'lr': 1e-6},
    {'params': chain(model.rel_classifier.parameters(), model.obj_1_classifier.parameters(), model.obj_2_classifier.parameters()), 'lr': 1e-4},
    {'params': chain(model.class_conv.parameters(), model.bounding_boxes_map.parameters()), 'lr': 1e-4},
])

# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Set up the gradient scaler for mixed precision training
scaler = GradScaler()

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch)

print("Training complete.")
