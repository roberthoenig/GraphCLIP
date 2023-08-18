import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
import os
from tempfile import NamedTemporaryFile
import numpy as np

def get_all_free_gpus_ids(min_mem=23000):
    try:
        with NamedTemporaryFile() as f:
            os.system(f"nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > {f.name}")
            memory_available = [int(x.split()[2]) for x in open(f.name, 'r').readlines()]
        if max(memory_available) < min_mem:
            print("Could not get any free GPU, probably results in a crash")
            return []
        return [i for i, mem in enumerate(memory_available) if mem > min_mem]
    except:
        print("Could not get any free GPU, probably results in a crash")
        return []

def get_free_gpu(min_mem=9000):
    try:
        with NamedTemporaryFile() as f:
            os.system(f"nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > {f.name}")
            memory_available = [int(x.split()[2]) for x in open(f.name, 'r').readlines()]
        if max(memory_available) < min_mem:
            print("Not enough memory on GPU, using CPU")
            return torch.device("cpu")
        return torch.device("cuda", np.argmax(memory_available))
    except:
        print("Could not get free GPU, using CPU")
        return torch.device("cpu")

def get_all_free_gpus_ids(min_mem=23000):
    try:
        with NamedTemporaryFile() as f:
            os.system(f"nvidia-smi -q -d Memory | grep -A5 GPU | grep Free > {f.name}")
            memory_available = [int(x.split()[2]) for x in open(f.name, 'r').readlines()]
        if max(memory_available) < min_mem:
            print("Could not get any free GPU, probably results in a crash")
            return []
        return [i for i, mem in enumerate(memory_available) if mem > min_mem]
    except:
        print("Could not get any free GPU, probably results in a crash")
        return []

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch):

    model.train()
    print("-" * 10)

    running_loss = 0.0
    running_corrects = 0

    # Iterate over the dataloader
    for batch_idx, (inputs, bounding_boxes, lrels, lobj1s, lobj2s) in enumerate(tqdm(dataloader)):
        inputs = inputs.to(device)
        bounding_boxes = bounding_boxes.to(device)
        lrels = lrels.to(device)
        lobj1s = lobj1s.to(device)
        lobj2s = lobj2s.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass with autocasting to enable mixed precision training
        # with autocast():
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16) as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable:
            outputs = model(inputs, bounding_boxes)
            rel, obj_1, obj_2 = outputs
            _, rel_preds = torch.max(rel, 1)
            _, obj_1_preds = torch.max(obj_1, 1)
            _, obj_2_preds = torch.max(obj_2, 1)
            loss_rel = criterion(rel, lrels)
            loss_obj_1 = criterion(obj_1, lobj1s)
            loss_obj_2 = criterion(obj_2, lobj2s)
            loss = loss_rel + loss_obj_1 + loss_obj_2

        # Backward pass and optimization step with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update running loss and correct predictions
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(rel_preds == lrels.data)

        # Compute the batch loss and accuracy
        batch_loss = loss.item()
        rel_acc = torch.sum(rel_preds == lrels.data).double() / inputs.size(0)
        obj_1_acc = torch.sum(obj_1_preds == lobj1s.data).double() / inputs.size(0)
        obj_2_acc = torch.sum(obj_2_preds == lobj2s.data).double() / inputs.size(0)

        # Log the batch loss and accuracy to wandb
        wandb.log({
            "batch_loss": batch_loss,
            "rel_loss": loss_rel.item(),
            "obj_1_loss": loss_obj_1.item(),
            "obj_2_loss": loss_obj_2.item(),
            "rel_ac": rel_acc.item(),
            "obj_1_acc": obj_1_acc.item(),
            "obj_2_acc": obj_2_acc.item(),
            "epoch": epoch,
            "epoch_progress": batch_idx/len(dataloader),
        })

    # Compute the epoch loss and accuracy
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    print(f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")




def evaluate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss_rel = 0.0
    running_loss_obj_1 = 0.0
    running_loss_obj_2 = 0.0
    running_corrects_rel = 0
    running_corrects_obj_1 = 0
    running_corrects_obj_2 = 0

    with torch.no_grad():
        for inputs, bounding_boxes, lrels, lobj1s, lobj2s in tqdm(dataloader):
            inputs = inputs.to(device)
            bounding_boxes = bounding_boxes.to(device)
            lrels = lrels.to(device)
            lobj1s = lobj1s.to(device)
            lobj2s = lobj2s.to(device)

            outputs = model(inputs, bounding_boxes)
            rel, obj_1, obj_2 = outputs
            _, rel_preds = torch.max(rel, 1)
            _ , obj_1_preds = torch.max(obj_1, 1)
            _ , obj_2_preds = torch.max(obj_2, 1)
            loss_rel = criterion(rel, lrels)
            loss_obj_1 = criterion(obj_1, lobj1s)
            loss_obj_2 = criterion(obj_2, lobj2s)

            running_loss_rel += loss_rel.item() * inputs.size(0)
            running_loss_obj_1 += loss_obj_1.item() * inputs.size(0)
            running_loss_obj_2 += loss_obj_2.item() * inputs.size(0)
            running_corrects_rel += torch.sum(rel_preds == lrels.data)
            running_corrects_obj_1 += torch.sum(obj_1_preds == lobj1s.data)
            running_corrects_obj_2 += torch.sum(obj_2_preds == lobj2s.data)

    val_loss_rel = running_loss_rel / len(dataloader.dataset)
    val_loss_obj_1 = running_loss_obj_1 / len(dataloader.dataset)
    val_loss_obj_2 = running_loss_obj_2 / len(dataloader.dataset)
    val_acc_rel = running_corrects_rel.double() / len(dataloader.dataset)
    val_acc_obj_1 = running_corrects_obj_1.double() / len(dataloader.dataset)
    val_acc_obj_2 = running_corrects_obj_2.double() / len(dataloader.dataset)
    

    wandb.log({
        "val_loss_rel": val_loss_rel,
        "val_loss_obj_1": val_loss_obj_1,
        "val_loss_obj_2": val_loss_obj_2, 
        "val_rel_acc": val_acc_rel.item(),
        "val_obj_1_acc": val_acc_obj_1.item(),
        "val_obj_2_acc": val_acc_obj_2.item(),
    })

    return val_acc_rel, val_acc_obj_1, val_acc_obj_2
