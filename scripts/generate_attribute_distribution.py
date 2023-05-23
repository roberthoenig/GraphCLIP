import json
from collections import defaultdict
import open_clip
import torch
import numpy as np
from pprint import pprint
from tqdm import tqdm

def main():
    # Load the JSON file
    with open('datasets/visual_genome/raw/scene_graphs.json', 'r') as f:
        data = json.load(f)
    print("scene_graphs.json loaded")

    def clip_embedding_txt_enc(txts):
        with torch.no_grad():
            tokens = tokenizer(txts)
            tokens[tokens == 49407] = 0
            tokens = tokens[:, 1:3]
            out = tokens.cpu()
            return out  

    for ending, model_name in [("_local","RN50"), ("", "ViT-g-14")]:
        tokenizer = open_clip.get_tokenizer(model_name=model_name)
        # Initialize the attribute distribution dictionary
        attribute_distribution = defaultdict(lambda: defaultdict(int))

        # Iterate over the data
        for entry in tqdm(data):
            for obj in entry['objects']:
                names_embed = clip_embedding_txt_enc(obj['names']).tolist()
                attributes_embed = clip_embedding_txt_enc(obj.get('attributes', [])).tolist()
                for name_embed in names_embed:
                    name_embed = tuple(name_embed)  # Convert list to tuple so it can be used as a dict key
                    for attribute_embed in attributes_embed:
                        attribute_embed = tuple(attribute_embed)  # Convert list to tuple so it can be used as a dict key
                        attribute_distribution[name_embed][attribute_embed] += 1

        # Convert defaultdicts to regular dicts for JSON serialization
        attribute_distribution = {k: dict(v) for k, v in attribute_distribution.items()}

        # Sort the attributes by their counts
        for name, attributes in tqdm(attribute_distribution.items()):
            attribute_distribution[name] = sorted(attributes.items(), key=lambda x: x[1], reverse=True)
            freqs = np.array([a[1] for a in attribute_distribution[name]])
            s = sum(freqs)
            probs = freqs / s
            attrs = [torch.tensor(list(a[0])) for a in attribute_distribution[name]]
            attribute_distribution[name] = {'probs': probs, 'attrs': attrs}

        # Save the attribute distribution dictionary
        OUT_PATH = f'datasets/visual_genome/raw/attribute_distribution{ending}.pt'
        torch.save(attribute_distribution, OUT_PATH)
            
if __name__ == "__main__":
    main() 