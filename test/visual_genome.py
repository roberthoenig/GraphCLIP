from datasets.visual_genome import VisualGenome

root = "datasets/visual_genome"
enc_cfg = {
    "model_name": "RN50",
    "pretrained": "openai",
    "device":  "cuda:0",
}
n_samples_process = 2

dataset = VisualGenome(root=root, enc_cfg=enc_cfg, n_samples_process=100)

print(dataset[0].x.shape)
print(dataset[0].edge_attr.shape)
print(dataset[0].edge_index.shape)
print(dataset[0].y.shape)

print(dataset[0].edge_index)
