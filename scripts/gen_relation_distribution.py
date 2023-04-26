import json
from pprint import pprint
import torch
from collections import defaultdict

def compute_histogram(words_list):
    # Count the frequency of each word in the list
    word_counts = defaultdict(int)
    for word in words_list:
        word_counts[word] += 1
    words = []
    probs = []
    w_p = sorted(list(word_counts.items()), key=lambda x: x[1], reverse=True)
    total_words = len(words_list)
    for word, count in w_p:
        words.append(word)
        probs.append(count / total_words)
    return {"words": words, "probs": probs}

def main():
    with open('datasets/visual_genome/raw/scene_graphs.json', 'r') as f:
        scene_graphs_dict = json.load(f)
    relationships = []
    for d in scene_graphs_dict:
        relationships += [r['predicate'] for r in d['relationships']]
    out = compute_histogram(relationships)
    with open("datasets/visual_genome/raw/relation_distribution.json", 'w') as f:
        json.dump(out, f)
        
if __name__ == '__main__':
    main()