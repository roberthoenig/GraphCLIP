import torch
import torch.nn as nn
from . import create_model_and_transforms

class ViT_RelClassifier(torch.nn.Module):
    def __init__(self, n_rel_classes, n_obj_classes, clip_model='ViT-B/32', pretrained='laion400m_e32'):
        super().__init__()
        model_vit, _, preprocess = create_model_and_transforms(clip_model, pretrained=pretrained)
        # self.preprocess = preprocess
        self.ViT = model_vit.visual
        self.vit_output_dim = self.ViT.output_dim
        self.rel_classifier = torch.nn.Linear(self.vit_output_dim, n_rel_classes)
        self.obj_1_classifier = torch.nn.Linear(self.vit_output_dim, n_obj_classes)
        self.obj_2_classifier = torch.nn.Linear(self.vit_output_dim, n_obj_classes)

        self.class_conv = nn.Conv1d(1,1,1)
        with torch.no_grad():
            self.class_conv.weight[0][0][0] = 0.0
            self.class_conv.bias[0] = 0.0
        self.bounding_boxes_map = nn.Linear(8,self.ViT.class_embedding_width)
        
    def forward(self, x, bounding_boxes = torch.zeros(1,8)):
        bounding_boxes_embedding = self.class_conv(self.bounding_boxes_map(bounding_boxes))
        x = self.ViT(x, bounding_boxes_embedding)
        rel = self.rel_classifier(x)
        obj_1 = self.obj_1_classifier(x)
        obj_2 = self.obj_2_classifier(x)
        return rel, obj_1, obj_2