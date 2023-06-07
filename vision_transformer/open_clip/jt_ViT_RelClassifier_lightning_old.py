import pytorch_lightning as pl
import torch
import torch.nn as nn
from itertools import chain
from . import create_model_and_transforms
import torchmetrics
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import Precision, Recall
# import wandb


class ViT_RelClassifier(pl.LightningModule):
    def __init__(self, 
                n_rel_classes, 
                n_obj_classes, 
                n_attr_classes, 
                clip_model='ViT-B/32', 
                pretrained='laion400m_e32', 
                dataloader_length=1,
                rel_occurence_probabilities = None,
                obj_occurence_probabilities = None,
                attr_occurence_probabilities = None,
                shallow=True,
                mode="bounding_boxes",
                with_object_heads=True,
        ):
        super().__init__()
        self.save_hyperparameters()
        model_vit, _, preprocess = create_model_and_transforms(clip_model, pretrained=pretrained)
        self.preprocess = preprocess
        self.ViT = model_vit.visual
        self.vit_output_dim = self.ViT.output_dim
        self.rel_classifier = self._make_classification_head(n_rel_classes, shallow) #torch.nn.Linear(self.vit_output_dim, n_rel_classes)
        self.obj_1_classifier = self._make_classification_head(n_obj_classes, shallow) #torch.nn.Linear(self.vit_output_dim, n_obj_classes)
        self.obj_2_classifier = self._make_classification_head(n_obj_classes, shallow) #torch.nn.Linear(self.vit_output_dim, n_obj_classes)
        self.attribute_classifier = self._make_classification_head(n_attr_classes, shallow) #torch.nn.Linear(self.vit_output_dim, 2)
        self.dataloader_length = dataloader_length

        self.class_conv = nn.Conv1d(1,1,1)
        with torch.no_grad():
            self.class_conv.weight[0][0][0] = 0.0
            self.class_conv.bias[0] = 0.0
        if mode == "bounding_boxes":
            # print("Using bounding boxes as input to the model.")
            self.bounding_boxes_map = nn.Linear(8, self.ViT.class_embedding_width)
        elif mode == "text_embeddings":
            # print("Using text embeddings as input to the model.")
            self.bounding_boxes_map = nn.Linear(2560, self.ViT.class_embedding_width)
        self.rel_accuracy = Accuracy(task="multiclass", num_classes=n_rel_classes)
        self.obj_1_accuracy = Accuracy(task="multiclass", num_classes=n_obj_classes)
        self.obj_2_accuracy = Accuracy(task="multiclass", num_classes=n_obj_classes)
        self.attr_precision = Precision(task="multilabel", num_labels=n_attr_classes) # always assumes two classes. Note that in case it's undefined, it is set to 0
        self.attr_recall = Recall(task="multilabel", num_labels=n_attr_classes)
        self.attr_accuracy = Accuracy(task="multilabel", num_labels=n_attr_classes)

        if rel_occurence_probabilities is not None and obj_occurence_probabilities is not None and attr_occurence_probabilities is not None:
            self.register_occurence_probabilities(rel_occurence_probabilities, obj_occurence_probabilities, attr_occurence_probabilities)
        else:
            # warn that the occurence probabilities are not registered
            self.register_occurence_probabilities(None, None, torch.ones(n_attr_classes,2)/2)
            # print("WARNING: occurence probabilities are not registered yet, but attributes are assumed to work better with a class weighted loss.")



    def register_occurence_probabilities(self, rel_occurence_probabilities, obj_occurence_probabilities, attr_occurence_probabilities, logger=None):
        self.rel_criterion = nn.CrossEntropyLoss(weight=self._calculate_class_weights(rel_occurence_probabilities))
        self.obj_criterion = nn.CrossEntropyLoss(weight=self._calculate_class_weights(obj_occurence_probabilities))
        attribute_binary_weights = torch.stack([ self._calculate_class_weights(attr_occ_p) for attr_occ_p in attr_occurence_probabilities ])
        attr_class_weights = attribute_binary_weights[:,0] # take the 0 class as default as the pos_weight only affects the 1 class
        attr_pos_weights = attribute_binary_weights[:,1]/(attribute_binary_weights[:,0]+1e-5)
        self.attr_criterion = nn.BCEWithLogitsLoss(weight=attr_class_weights, pos_weight=attr_pos_weights)
        # print("Attribute class weights: ", attr_class_weights)
        # print("Attribute pos weights: ", attr_pos_weights)
        if logger is not None:
            t0 = wandb.Table(columns =["class 0","class 1"] , data=attr_occurence_probabilities.cpu().numpy())
            t1 = wandb.Table(columns =["attr class weights"] , data=attr_class_weights.reshape(-1,1).cpu().numpy())
            t2 = wandb.Table(columns =["attr pos weights"] , data=attr_pos_weights.reshape(-1,1).cpu().numpy())
            logger.experiment.log({"attribute occ probs": t0})
            logger.experiment.log({"attribute class weights": t1})
            logger.experiment.log({"attribute pos weights": t2})
            print("attr occ probs: ", attr_occurence_probabilities, "\nattr class weights: ", attr_class_weights, "\nattr pos weights: ", attr_pos_weights)
            print("occurence probabilities registered.")

    def _make_classification_head(self, n_classes, shallow=False):
        if shallow:
            return nn.Linear(self.vit_output_dim, n_classes)
        layer1 = nn.Linear(self.vit_output_dim, n_classes*2)
        layer2 = nn.Linear(n_classes*2, n_classes)
        return nn.Sequential(layer1, nn.ReLU(), layer2)
    
    def _calculate_class_weights(self, occurence_probabilities):
        if occurence_probabilities is None:
            return None
        # normalize weights such that in expectation for each class the weight is 1/n, so the expected overall weight is 1
        weights = 1 / (occurence_probabilities+1e-5)
        weights = weights / len(weights)
        return weights
        
    def forward(self, x, bounding_boxes=torch.zeros(1, 8)):
        bounding_boxes_embedding = self.class_conv(self.bounding_boxes_map(bounding_boxes))
        x = self.ViT(x, bounding_boxes_embedding)
        rel = self.rel_classifier(x)
        obj_1 = self.obj_1_classifier(x)
        obj_2 = self.obj_2_classifier(x)
        attr = self.attribute_classifier(x)
        return rel, obj_1, obj_2, attr

    def training_step(self, batch, batch_idx):
        inputs, bounding_boxes, lrels, lobj1s, lobj2s, lattr, rel_mask, attr_mask = batch
        rel, obj_1, obj_2, attr = self(inputs, bounding_boxes)
        loss_obj_1 = self.obj_criterion(obj_1, lobj1s)
        loss_obj_2 = self.obj_criterion(obj_2, lobj2s)
        loss = loss_obj_1 + loss_obj_2
        if len(rel[rel_mask.view(-1)==1]) > 0:
            loss_rel = self.rel_criterion(rel[rel_mask.view(-1)==1], lrels[rel_mask.view(-1)==1])
            loss += loss_rel
        if len(attr_mask.view(-1)==1) > 0:
            loss_attr = self.attr_criterion(attr[attr_mask.view(-1)==1], lattr[attr_mask.view(-1)==1])
            loss += 10*loss_attr

        self.log_dict({
            "train_loss": loss.item(),
            "rel_loss": loss_rel.item() if len(rel[rel_mask.view(-1)==1]) > 0 else 0.0,
            "obj_1_loss": loss_obj_1.item(),
            "obj_2_loss": loss_obj_2.item(),
            "attr_loss": loss_attr.item() if len(attr_mask.view(-1)==1) > 0 else 0.0,
            "rel_acc": self.rel_accuracy(rel[rel_mask.view(-1)==1], lrels[rel_mask.view(-1)==1]) if len(rel[rel_mask.view(-1)==1]) > 0 else 0.0,
            "obj_1_acc": self.obj_1_accuracy(obj_1, lobj1s),
            "obj_2_acc": self.obj_2_accuracy(obj_2, lobj2s),
            "attr_precision": self.attr_precision(attr, lattr) if len(attr_mask.view(-1)==1) > 0 else 0.0,
            "attr_recall": self.attr_recall(attr, lattr) if len(attr_mask.view(-1)==1) > 0 else 0.0,
            "attr_acc": self.attr_accuracy(attr, lattr) if len(attr_mask.view(-1)==1) > 0 else 0.0,
            "epoch": float(self.current_epoch),
            "epoch_progress": batch_idx/self.dataloader_length
        }, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss
    
    # def training_step_end(self, outputs):
    #     # Compute the batch loss and accuracy
    #     self.rel_accuracy.update(outputs["rel"], outputs["lrels"])
    #     self.obj_1_accuracy.update(outputs["obj_1"], outputs["lobj1s"])
    #     self.obj_2_accuracy.update(outputs["obj_2"], outputs["lobj2s"])

    #     # Log the batch loss and accuracy to wandb
    #     self.log("train_loss", outputs["loss"].item())
    #     self.log("rel_loss", outputs["loss_rel"].item())
    #     self.log("obj_1_loss", outputs["loss_obj_1"].item())
    #     self.log("obj_2_loss", outputs["loss_obj_2"].item())
    #     self.log("rel_ac", self.rel_accuracy)
    #     self.log("obj_1_acc", self.obj_1_accuracy)
    #     self.log("obj_2_acc", self.obj_2_accuracy)
    #     self.log("epoch", self.current_epoch)
    #     self.log("epoch_progress", outputs["batch_idx"]/len(self.dataloader_length))


    def validation_step(self, batch, batch_idx):
        inputs, bounding_boxes, lrels, lobj1s, lobj2s, lattr, rel_mask, attr_mask = batch
        rel, obj_1, obj_2, attr = self(inputs, bounding_boxes)
        loss_obj_1 = self.obj_criterion(obj_1, lobj1s)
        loss_obj_2 = self.obj_criterion(obj_2, lobj2s)
        loss = loss_obj_1 + loss_obj_2
        if len(rel[rel_mask.view(-1)==1]) > 0:
            loss_rel = self.rel_criterion(rel[rel_mask.view(-1)==1], lrels[rel_mask.view(-1)==1])
            loss += loss_rel
            self.rel_accuracy(rel[rel_mask.view(-1)==1], lrels[rel_mask.view(-1)==1])
        if len(attr_mask.view(-1)==1) > 0:
            loss_attr = self.attr_criterion(attr[attr_mask.view(-1)==1], lattr[attr_mask.view(-1)==1])
            loss += 10*loss_attr
            self.attr_precision(attr, lattr)
            self.attr_recall(attr, lattr)
            self.attr_accuracy(attr, lattr)
        self.obj_1_accuracy(obj_1, lobj1s)
        self.obj_2_accuracy(obj_2, lobj2s)
        return loss
        # return {"loss": loss, "rel": rel, "obj_1": obj_1, "obj_2": obj_2, "attr": attr, "lrels": lrels, "lobj1s": lobj1s, "lobj2s": lobj2s, "lattr": lattr, "batch_idx": batch_idx}
    
    # def validation_step_end(self, outputs):
    #     # self.log("val_loss", outputs["loss"])
    #     # Compute the batch loss and accuracy
    #     self.rel_accuracy.update(outputs["rel"], outputs["lrels"])
    #     self.obj_1_accuracy.update(outputs["obj_1"], outputs["lobj1s"])
    #     self.obj_2_accuracy.update(outputs["obj_2"], outputs["lobj2s"])
    #     return outputs["loss"]
    
    def on_validation_epoch_end(self):
        # Log the batch loss and accuracy to wandb
        self.log("val_acc", self.rel_accuracy.compute(), sync_dist=True)
        self.log("val_rel_acc", self.rel_accuracy.compute(), sync_dist=True)
        self.log("val_obj_1_acc", self.obj_1_accuracy.compute(), sync_dist=True)
        self.log("val_obj_2_acc", self.obj_2_accuracy.compute(), sync_dist=True)
        self.log("val_attr_precision", self.attr_precision.compute(), sync_dist=True)
        self.log("val_attr_recall", self.attr_recall.compute(), sync_dist=True)
        self.log("val_attr_accuracy", self.attr_accuracy.compute(), sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.ViT.parameters(), 'lr': 1e-6},
            {'params': chain(self.rel_classifier.parameters(), self.obj_1_classifier.parameters(), self.obj_2_classifier.parameters()), 'lr': 1e-4},
            {'params': chain(self.class_conv.parameters(), self.bounding_boxes_map.parameters()), 'lr': 1e-4},
        ])
        return optimizer

    # def criterion(self, y_hat, y):
    #     return torch.nn.CrossEntropyLoss()(y_hat, y)
