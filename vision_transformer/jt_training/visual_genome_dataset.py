import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sys
from os.path import dirname, abspath
d = dirname(dirname(dirname(abspath(__file__))))
sys.path.append(d)
from datasets.VG_graphs import get_filtered_relationships, get_filtered_objects, get_filtered_attributes, copy_graph, get_realistic_graphs_dataset, plot_graph

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
        return self.getitem_from_id_edge(image_id, edge)


    def getitem_from_id_edge(self, image_id, edge):
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
    train_size = int(0.8 * len(filtered_graphs))
    filtered_graphs_train, filtered_graphs_val = torch.utils.data.random_split(filtered_graphs, [train_size, len(filtered_graphs) - train_size])
    print("Done loading filtered graphs.")
    id_edge_graph_dict_train = {
        (g.image_id, e
        ): g
        for g in filtered_graphs_train for e in g.edges()
    }
    id_edge_graph_dict_val = {
        (g.image_id, e
        ): g
        for g in filtered_graphs_val for e in g.edges()
    }
    dataset_train = CustomImageDataset(image_dir, id_edge_graph_dict_train, preprocess_func)
    dataset_val = CustomImageDataset(image_dir, id_edge_graph_dict_val, preprocess_func)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return dataloader_train, dataloader_val


def get_realistic_graphs_dataset_ViT(
        preprocess_func,
        image_dir="/local/home/stuff/visual-genome/VG/"
):
    dataset = get_realistic_graphs_dataset()
    id_edge_graph_dict_test_orig = {
        (d['original_graph'].image_id, d['changed_edge']
        ): d['original_graph']
        for d in dataset
    }
    id_edge_graph_dict_test_adv = {
        (d['adv_graph'].image_id, d['changed_edge']
        ): d['adv_graph']
        for d in dataset
    }
    dataset_test_orig = CustomImageDataset(image_dir, id_edge_graph_dict_test_orig, preprocess_func)
    dataset_test_adv = CustomImageDataset(image_dir, id_edge_graph_dict_test_adv, preprocess_func)
    return dataset_test_orig, dataset_test_adv, dataset

