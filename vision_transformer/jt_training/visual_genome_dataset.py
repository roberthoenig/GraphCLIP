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
import random

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, id_edge_graph_dict, preprocess_func, mode, compute_occurence_probabilities=True):
        self.mode = mode
        self.image_dir = image_dir
        self.id_edge_graph_dict = id_edge_graph_dict
        self.preprocess_func = preprocess_func
        self.rel_classes = {rel:i for i,rel in enumerate(get_filtered_relationships())}
        self.obj_classes = {obj:i for i,obj in enumerate(get_filtered_objects())}
        self.attr_classes = {attr:i for i,attr in enumerate(get_filtered_attributes())}
        objs = get_filtered_objects()
        obj_embeddings = torch.load("/local/home/jthomm/GraphCLIP/datasets/visual_genome/processed/filtered_object_label_embeddings.pt")
        self.text_embeddings = {obj:torch.tensor(obj_embeddings[i]) for i,obj in enumerate(objs)}
        if compute_occurence_probabilities:
            rel_occurence_probabilities = torch.zeros(len(self.rel_classes))
            obj_occurence_probabilities = torch.zeros(len(self.obj_classes))
            attr_occurence_probabilities = torch.zeros(len(self.attr_classes))
            rel_divisor = 0
            obj_divisor = 0
            attr_divisor = 0
            for image_id, edge in self.id_edge_graph_dict.keys():
                g = self.id_edge_graph_dict[(image_id, edge)]
                # if edge[0] == edge[1]:
                #     for attr in g.nodes[edge[0]]['attributes']:
                #         attr_occurence_probabilities[self.attr_classes[attr]] += 1
                #     obj_occurence_probabilities[self.obj_classes[g.nodes[edge[0]]['name']]] += 1
                # else:
                if edge[0] != edge[1]:
                    rel_occurence_probabilities[self.rel_classes[g.edges[edge]['predicate']]] += 1
                    rel_divisor += 1
                obj_occurence_probabilities[self.obj_classes[g.nodes[edge[0]]['name']]] += 1
                obj_occurence_probabilities[self.obj_classes[g.nodes[edge[1]]['name']]] += 1
                obj_divisor += 2
                for attr in g.nodes[edge[0]]['attributes']:
                    attr_occurence_probabilities[self.attr_classes[attr]] += 1
                if len(g.nodes[edge[0]]['attributes']) > 0:
                    attr_divisor += 1
                # for attr in g.nodes[edge[1]]['attributes']:
                #     attr_occurence_probabilities[self.attr_classes[attr]] += 1
            self.rel_occurence_probabilities = rel_occurence_probabilities / rel_divisor
            self.obj_occurence_probabilities = obj_occurence_probabilities / obj_divisor
            attr_occ_p = attr_occurence_probabilities / attr_divisor
            self.attr_occurence_probabilities = torch.zeros(len(self.attr_classes),2)
            self.attr_occurence_probabilities[:,1] = attr_occ_p
            self.attr_occurence_probabilities[:,0] = 1 - attr_occ_p

        self.image_ids_edges = list(self.id_edge_graph_dict.keys())

    def __len__(self):
        return len(self.image_ids_edges)

    def __getitem__(self, idx):
        image_id, edge = self.image_ids_edges[idx]
        return self.getitem_from_id_edge(image_id, edge, mode = self.mode)

    def get_bounding_boxes(self,g, edge, image_width, image_height):
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
        return bounding_boxes

    def getitem_from_id_edge(self, image_id, edge, mode='bounding_boxes'):
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        g = self.id_edge_graph_dict[(image_id, edge)]
        if mode == 'bounding_boxes':
            image_width, image_height = image.size
            bounding_boxes = self.get_bounding_boxes(g, edge, image_width, image_height)
        elif mode == 'text_embeddings':
            text_embd_obj1 = self.text_embeddings[g.nodes[edge[0]]['name']]
            text_embd_obj2 = self.text_embeddings[g.nodes[edge[1]]['name']]
            full_text_clip_embd = torch.cat((text_embd_obj1, text_embd_obj2), dim=0).reshape(1,-1)

        image = self.preprocess_func(image)
        
        if edge[0] == edge[1]:
            # print("case 1")
            rel_label = torch.tensor(0)
            rel_mask = torch.tensor(0)
        else:
            # print("case 2")
            rel_label = torch.tensor(self.rel_classes[g.edges[edge]['predicate']])
            rel_mask = torch.tensor(1)
        # print("1.", g.nodes[edge[0]]['name'], "2.", self.obj_classes[g.nodes[edge[0]]['name']], "3.", torch.tensor(self.obj_classes[g.nodes[edge[0]]['name']]))
        obj1_label = torch.tensor(self.obj_classes[g.nodes[edge[0]]['name']])
        obj2_label = torch.tensor(self.obj_classes[g.nodes[edge[1]]['name']])
        # print("obj1_label", obj1_label, "obj2_label", obj2_label, "obj1_label.shape", obj1_label.shape, "obj2_label.shape", obj2_label.shape)

        # try:
        positive_attr_classes = [self.attr_classes[attr] for attr in g.nodes[edge[0]]['attributes']]
        if len(positive_attr_classes) == 0:
            attr_mask = torch.tensor(0)
        else:
            attr_mask = torch.tensor(1)
        attr_label = torch.zeros(len(self.attr_classes))
        attr_label[positive_attr_classes] = 1
        # except:
        #     attr_mask = torch.tensor(0)
        #     attr_label = torch.zeros(len(self.attr_classes))

        # print('image', image.shape, 'bounding_boxes', bounding_boxes.shape, 'rel_label', rel_label.shape, 'obj1_label', obj1_label.shape, 'obj2_label', obj2_label.shape, 'attr_label', attr_label.shape, 'rel_mask', rel_mask.shape)
        return image, bounding_boxes if mode == 'bounding_boxes' else full_text_clip_embd, rel_label, obj1_label, obj2_label, attr_label, rel_mask, attr_mask


def filter_out_adversarial_dataset(filtered_graphs):
    adv_dataset = get_realistic_graphs_dataset()
    image_ids = set([d['original_graph'].image_id for d in adv_dataset])
    filtered_graphs = [g for g in filtered_graphs if g.image_id not in image_ids]
    return filtered_graphs


def get_dataloader( 
        preprocess_func,
        filtered_graphs_path="/local/home/jthomm/GraphCLIP/datasets/visual_genome/processed/", 
        image_dir="/local/home/stuff/visual-genome/VG/",
        mode='bounding_boxes',
        batch_size=64, 
        num_workers=8, 
        shuffle=True,
        testing_only=False,
    ):
    print("Loading filtered graphs...")
    if testing_only:
        filtered_graphs = torch.load(filtered_graphs_path + "filtered_graphs_test_small.pt") # much faster to load
    else:
        filtered_graphs = torch.load(filtered_graphs_path + "filtered_graphs.pt")
    filtered_graphs = filter_out_adversarial_dataset(filtered_graphs)
    train_size = int(0.8 * len(filtered_graphs))
    filtered_graphs_train, filtered_graphs_val = torch.utils.data.random_split(filtered_graphs, [train_size, len(filtered_graphs) - train_size], generator=torch.Generator().manual_seed(42032))
    print("Done loading filtered graphs.")
    id_edge_graph_dict_train = {
        (g.image_id, e
        ): g
        for g in filtered_graphs_train for e in g.edges()
    }
    id_node_graph_dict_train = {
        (g.image_id, (n,n)
        ): g
        for g in filtered_graphs_train for n in g.nodes() if len(g.nodes[n]['attributes']) > 0 # only use nodes with attributes
    }
    # get a random subset of keys and extend the dict with them
    subset = random.sample(list(id_node_graph_dict_train.keys()), len(id_edge_graph_dict_train)//4) # sample is without replacement
    for k in subset:
        id_edge_graph_dict_train[k] = id_node_graph_dict_train[k]
    id_edge_graph_dict_val = {
        (g.image_id, e
        ): g
        for g in filtered_graphs_val for e in g.edges()
    }
    id_node_graph_dict_val = {
        (g.image_id, (n,n)
        ): g
        for g in filtered_graphs_val for n in g.nodes() if len(g.nodes[n]['attributes']) > 0 # only use nodes with attributes
    }
    # get a random subset of keys and extend the dict with them
    subset = random.sample(list(id_node_graph_dict_val.keys()), len(id_edge_graph_dict_val)//4)
    for k in subset:
        id_edge_graph_dict_val[k] = id_node_graph_dict_val[k]
    dataset_train = CustomImageDataset(image_dir, id_edge_graph_dict_train, preprocess_func, mode=mode)
    dataset_val = CustomImageDataset(image_dir, id_edge_graph_dict_val, preprocess_func, mode=mode)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return dataloader_train, dataloader_val


def get_realistic_graphs_dataset_ViT(
        preprocess_func,
        image_dir="/local/home/stuff/visual-genome/VG/",
        mode='bounding_boxes',
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
    dataset_test_orig = CustomImageDataset(image_dir, id_edge_graph_dict_test_orig, preprocess_func, mode=mode, compute_occurence_probabilities=False)
    dataset_test_adv = CustomImageDataset(image_dir, id_edge_graph_dict_test_adv, preprocess_func, mode=mode, compute_occurence_probabilities=False)
    return dataset_test_orig, dataset_test_adv, dataset

