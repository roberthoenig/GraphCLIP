import pytorch_lightning as pl
import torch
import torch.nn as nn
from itertools import chain
from open_clip import create_model_and_transforms
import torchmetrics
from torchmetrics.classification.accuracy import Accuracy


class ViT_RelClassifier(pl.LightningModule):
    def __init__(self, n_rel_classes, n_obj_classes, clip_model='ViT-B/32', pretrained='laion400m_e32', dataloader_length=1):
        super().__init__()
        model_vit, _, preprocess = create_model_and_transforms(clip_model, pretrained=pretrained)
        self.preprocess = preprocess
        self.ViT = model_vit.visual
        self.vit_output_dim = self.ViT.output_dim
        self.rel_classifier = torch.nn.Linear(self.vit_output_dim, n_rel_classes)
        self.obj_1_classifier = torch.nn.Linear(self.vit_output_dim, n_obj_classes)
        self.obj_2_classifier = torch.nn.Linear(self.vit_output_dim, n_obj_classes)
        self.dataloader_length = dataloader_length

        self.class_conv = nn.Conv1d(1,1,1)
        with torch.no_grad():
            self.class_conv.weight[0][0][0] = 0.0
            self.class_conv.bias[0] = 0.0
        self.bounding_boxes_map = nn.Linear(8, self.ViT.class_embedding_width)
        self.rel_accuracy = Accuracy(task="multiclass", num_classes=n_rel_classes)
        self.obj_1_accuracy = Accuracy(task="multiclass", num_classes=n_obj_classes)
        self.obj_2_accuracy = Accuracy(task="multiclass", num_classes=n_obj_classes)
        
    def forward(self, x, bounding_boxes=torch.zeros(1, 8)):
        bounding_boxes_embedding = self.class_conv(self.bounding_boxes_map(bounding_boxes))
        x = self.ViT(x, bounding_boxes_embedding)
        rel = self.rel_classifier(x)
        obj_1 = self.obj_1_classifier(x)
        obj_2 = self.obj_2_classifier(x)
        return rel, obj_1, obj_2

    def training_step(self, batch, batch_idx):
        inputs, bounding_boxes, lrels, lobj1s, lobj2s = batch
        rel, obj_1, obj_2 = self(inputs, bounding_boxes)
        loss_rel = self.criterion(rel, lrels)
        loss_obj_1 = self.criterion(obj_1, lobj1s)
        loss_obj_2 = self.criterion(obj_2, lobj2s)
        loss = loss_rel + loss_obj_1 + loss_obj_2

        return {
            "loss": loss, "rel": rel, "obj_1": obj_1, "obj_2": obj_2, "lrels": lrels, "lobj1s": lobj1s, "lobj2s": lobj2s, "batch_idx": batch_idx,
            "loss_rel": loss_rel, "loss_obj_1": loss_obj_1, "loss_obj_2": loss_obj_2
        }
    
    def training_step_end(self, outputs):
        # Compute the batch loss and accuracy
        self.rel_accuracy.update(outputs["rel"], outputs["lrels"])
        self.obj_1_accuracy.update(outputs["obj_1"], outputs["lobj1s"])
        self.obj_2_accuracy.update(outputs["obj_2"], outputs["lobj2s"])

        # Log the batch loss and accuracy to wandb
        # self.log({
        #     "train_loss": outputs["loss"].item(),
        #     "rel_loss": outputs["loss_rel"].item(),
        #     "obj_1_loss": outputs["loss_obj_1"].item(),
        #     "obj_2_loss": outputs["loss_obj_2"].item(),
        #     "rel_ac": self.rel_accuracy,
        #     "obj_1_acc": self.obj_1_accuracy,
        #     "obj_2_acc": self.obj_2_accuracy,
        #     "epoch": self.current_epoch,
        #     "epoch_progress": outputs["batch_idx"]/len(self.dataloader_length),
        # })
        self.log("train_loss", outputs["loss"].item())
        self.log("rel_loss", outputs["loss_rel"].item())
        self.log("obj_1_loss", outputs["loss_obj_1"].item())
        self.log("obj_2_loss", outputs["loss_obj_2"].item())
        self.log("rel_ac", self.rel_accuracy)
        self.log("obj_1_acc", self.obj_1_accuracy)
        self.log("obj_2_acc", self.obj_2_accuracy)
        self.log("epoch", self.current_epoch)
        self.log("epoch_progress", outputs["batch_idx"]/len(self.dataloader_length))

    def validation_step(self, batch, batch_idx):
        inputs, bounding_boxes, lrels, lobj1s, lobj2s = batch
        rel, obj_1, obj_2 = self(inputs, bounding_boxes)
        loss_rel = self.criterion(rel, lrels)
        loss_obj_1 = self.criterion(obj_1, lobj1s)
        loss_obj_2 = self.criterion(obj_2, lobj2s)
        loss = loss_rel + loss_obj_1 + loss_obj_2
        return {"loss": loss, "rel": rel, "obj_1": obj_1, "obj_2": obj_2, "lrels": lrels, "lobj1s": lobj1s, "lobj2s": lobj2s, "batch_idx": batch_idx}
    
    def validation_step_end(self, outputs):
        # self.log("val_loss", outputs["loss"])
        # Compute the batch loss and accuracy
        self.rel_accuracy.update(outputs["rel"], outputs["lrels"])
        self.obj_1_accuracy.update(outputs["obj_1"], outputs["lobj1s"])
        self.obj_2_accuracy.update(outputs["obj_2"], outputs["lobj2s"])
        return outputs["loss"]
    
    def on_validation_epoch_end(self):
        # Log the batch loss and accuracy to wandb
        # self.log({
        #     'val_acc': self.rel_accuracy,
        #     "val_rel_ac": self.rel_accuracy,
        #     "val_obj_1_acc": self.obj_1_accuracy,
        #     "val_obj_2_acc": self.obj_2_accuracy,
        # })
        self.log("val_acc", self.rel_accuracy)
        self.log("val_rel_ac", self.rel_accuracy)
        self.log("val_obj_1_acc", self.obj_1_accuracy)
        self.log("val_obj_2_acc", self.obj_2_accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.ViT.parameters(), 'lr': 1e-6},
            {'params': chain(self.rel_classifier.parameters(), self.obj_1_classifier.parameters(), self.obj_2_classifier.parameters()), 'lr': 1e-4},
            {'params': chain(self.class_conv.parameters(), self.bounding_boxes_map.parameters()), 'lr': 1e-4},
        ])
        return optimizer

    def criterion(self, y_hat, y):
        return torch.nn.CrossEntropyLoss()(y_hat, y)
