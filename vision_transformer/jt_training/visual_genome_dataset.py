import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sys
from os.path import dirname, abspath
d = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(d)
from datasets.VG_graphs import get_filtered_relationships, get_filtered_objects, get_filtered_attributes

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, id_edge_graph_dict, preprocess_func):
        self.image_dir = image_dir
        self.id_edge_graph_dict = id_edge_graph_dict
        self.preprocess_func = preprocess_func
        self.rel_classes = {rel:i for i,rel in enumerate(get_filtered_relationships())}
        self.obj_classes = {obj:i for i,obj in enumerate(get_filtered_objects())}

        self.image_ids_edges = list(self.id_edge_graph_dict.keys())

    def __len__(self):
        return len(self.image_ids_edges)

    def __getitem__(self, idx):
        image_id, edge = self.image_ids_edges[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        g = self.id_edge_graph_dict[(image_id, edge)]
        image_width, image_height = image.size
        ax, ay, aw, ah, bx, by, bw, bh = (
            g.nodes[edge[0]]['x'],
            g.nodes[edge[0]]['y'],
            g.nodes[edge[0]]['w'],
            g.nodes[edge[0]]['h'],
            g.nodes[edge[1]]['x'],
            g.nodes[edge[1]]['y'],
            g.nodes[edge[1]]['w'],
            g.nodes[edge[1]]['h'],
        )
        bounding_boxes = torch.tensor([
            ax/image_width, 
            ay/image_height, 
            aw/image_width, 
            ah/image_height, 
            bx/image_width, 
            by/image_height, 
            bw/image_width, 
            bh/image_height
        ]).view(1,8)
        image = self.preprocess_func(image)
        

        rel_label = torch.tensor(self.rel_classes[g.edges[edge]['predicate']])
        obj1_label = torch.tensor(self.obj_classes[g.nodes[edge[0]]['name']])
        obj2_label = torch.tensor(self.obj_classes[g.nodes[edge[1]]['name']])

        return image, bounding_boxes, rel_label, obj1_label, obj2_label

def get_dataloader( 
        preprocess_func,
        filtered_graphs_path="/local/home/jthomm/GraphCLIP/datasets/visual_genome/processed/", 
        image_dir="/local/home/stuff/visual-genome/VG/",
        batch_size=64, 
        num_workers=4, 
        shuffle=True,
        testing_only=False
    ):
    print("Loading filtered graphs...")
    if testing_only:
        filtered_graphs = torch.load(filtered_graphs_path + "filtered_graphs_test_small.pt") # much faster to load
    else:
        filtered_graphs = torch.load(filtered_graphs_path + "filtered_graphs.pt")
    print("Done loading filtered graphs.")
    id_edge_graph_dict = {
        (g.image_id, e
        ): g
        for g in filtered_graphs for e in g.edges()
    }
    dataset = CustomImageDataset(image_dir, id_edge_graph_dict, preprocess_func)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return dataloader