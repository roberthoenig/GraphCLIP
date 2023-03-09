import torch
from PIL import Image
import open_clip
from datasets.mscoco import MSCOCO
from datasets.visual_genome import VisualGenome
from utils.eval_utils import compute_ranking_metrics_from_features
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader

class GNN():
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1024, 256)
        self.conv2 = GCNConv(256, 1024)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)

        return x

class GraphCLIP():
    def __init__(self, config):
        self.config = config
        
    def train(self):
        # Model
        model = GNN().to(self.config["device"])
        model.train()
        # Dataset
        if self.config["dataset"] == "VisualGenome":
            dataset = VisualGenome(**self.config["dataset_args"])
        else:
            raise Exception(f"Unkown dataset {self.config['dataset']}.")
        train_ratio = self.config["train_args"]["train_val_split"]
        train_set, val_set = torch.utils.data.random_split(dataset, [train_ratio, 1-train_ratio])
        train_dloader = DataLoader(train_set, batch_size=self.config["train_args"]["batch_size"], shuffle=True)
        val_dloader = DataLoader(val_set, batch_size=1, shuffle=False)
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["train_args"]["learning_rate"])
        # Training
        total_loss = total_examples = 0
        for data in train_dloader:
            data = data.to(self.config["device"])
            optimizer.zero_grad()
            loss = contrastive_loss(model(data.x, data.adj_t), data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_nodes
        return total_loss / total_examples
        
    def eval(self):
        # Model
        model_name = self.config["model_args"]["model_name"]
        pretrained = self.config["model_args"]["pretrained"]
        model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained, device=self.config["device"])
        tokenizer = open_clip.get_tokenizer(model_name=model_name)
        
        # Dataset
        if self.config["dataset"] == "MSCOCO":
            dataset = MSCOCO(**self.config["dataset_args"])
        else:
            raise Exception(f"Unkown dataset {self.config['dataset']}.")
        
        # Compute features
        with torch.no_grad(), torch.cuda.amp.autocast():
            # (n_samples, emb_sz) 
            print("Computing image embeddings.")
            img_features = torch.concat([model.encode_image(preprocess(Image.open(path)).to(self.config["device"]).unsqueeze(0)).cpu() for path in tqdm(dataset.img_paths)])
            # (n_samples, captions_per_image, emb_sz) 
            print("Computing caption embeddings.")
            cap_features = torch.stack([model.encode_text(tokenizer(captions).to(self.config["device"])).cpu() for captions in tqdm(dataset.captions)])
            img_features /= img_features.norm(dim=-1, keepdim=True)
            cap_features /= cap_features.norm(dim=-1, keepdim=True)

        # Compute metrics
        compute_ranking_metrics_from_features(
            img_features=img_features,
            cap_features=cap_features,
            ks=self.config['eval_args']['ks']
        )