import logging
from unittest import result
import torch
import pprint
from tqdm import tqdm

# recall at k (same as R precision with R=k)
# what percentage of relevant documents is included in the top-k predictions, on average?
def get_recall_at_k(ranks, k, relevant_idx, n_relevant):
    assert k <= len(ranks)
    count_relevant = (ranks[relevant_idx] < k).sum()
    percentage = count_relevant.item() / n_relevant
    return percentage

# precision at k
# what percentage of documents in the top-k predictions are relevant, on average?
def get_precision_at_k(ranks, k, relevant_idx):
    assert k <= len(ranks)
    count_relevant = (ranks[relevant_idx] < k).sum()
    percentage = count_relevant.item() / k
    return percentage

# img_features: (n_samples, emb_sz) 
# cap_features: (n_samples, captions_per_image, emb_sz) 
def compute_ranking_metrics_from_features(img_features, cap_features, ks):
    n_samples, captions_per_image, emb_sz = cap_features.shape
    result = dict()
    # Rank captions
    logging.info("Ranking captions...")
    ranked_features = cap_features
    other_features = img_features
    result["ranking_captions"] = dict()
    result["ranking_captions"]["recall@k"] = dict()
    result["ranking_captions"]["precision@k"] = dict()
    for k in ks:
        result["ranking_captions"]["recall@k"][k] = []
        result["ranking_captions"]["precision@k"][k] = []
    for idx in tqdm(range(n_samples)):
        scores = (other_features[idx].unsqueeze(0).unsqueeze(0) * ranked_features).sum(dim=-1)
        scores_shape = scores.shape
        ranks = torch.argsort(torch.argsort(scores.flatten(), descending=True)).reshape(scores_shape)
        for k in ks:
            r_at_k = get_recall_at_k(ranks, k, idx, captions_per_image)
            result["ranking_captions"]["recall@k"][k].append(r_at_k)
            p_at_k = get_precision_at_k(ranks, k, idx)
            result["ranking_captions"]["precision@k"][k].append(p_at_k)
    
    # Rank images
    logging.info("Ranking images...")
    ranked_features = img_features
    other_features = cap_features
    result["ranking_images"] = dict()
    result["ranking_images"]["recall@k"] = dict()
    result["ranking_images"]["precision@k"] = dict()
    for k in ks:
        result["ranking_images"]["recall@k"][k] = []
        result["ranking_images"]["precision@k"][k] = []
    for idx in tqdm(range(n_samples)):
        for c_idx in range(captions_per_image):
            scores = (other_features[idx][c_idx].unsqueeze(0) * ranked_features).sum(dim=-1)
            ranks = torch.argsort(torch.argsort(scores, descending=True))
            for k in ks:
                r_at_k = get_recall_at_k(ranks, k, idx, 1)
                result["ranking_images"]["recall@k"][k].append(r_at_k)
                p_at_k = get_precision_at_k(ranks, k, idx)
                result["ranking_images"]["precision@k"][k].append(p_at_k)
    
    def lists_to_mean(d):
        for k, v in d.items():
            if isinstance(v, dict):
                lists_to_mean(v) 
            else:
                d[k] = torch.mean(torch.tensor(v))
    lists_to_mean(result)
    logging.info("Result: " + pprint.pformat(result))
    return result

# img_features: (n_samples, emb_sz) 
# features_gt: (n_samples, emb_sz) 
# features_adv: (n_samples, emb_sz) 
def compute_accuracy_from_adversarial_features(img_features, features_gt, features_adv):
    scores_gt = (img_features * features_gt).sum(dim=-1)
    scores_adv = (img_features * features_adv).sum(dim=-1)
    is_correct = scores_gt > scores_adv
    acc = is_correct.float().mean()
    result = {
        "accuracy": f"{acc:.2f}"
    }
    logging.info("Result: " + pprint.pformat(result))