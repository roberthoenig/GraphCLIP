import torch
from torch.nn.functional import cross_entropy

# adapted from https://cv-tricks.com/how-to/understanding-clip-by-openai/
# y_pred: (batch_size, embedding_size)
# y_gt: (batch_size, embedding_size)
def contrastive_loss(y_pred, y_gt):
    y_pred = y_pred / torch.norm(y_pred, dim=1)
    y_gt = y_gt / torch.norm(y_gt, dim=1)
    logits = torch.matmul(y_pred, y_gt.transpose(0,1))
    labels = torch.arange(y_pred.shape[0]) 
    loss_1 = cross_entropy(input=logits, target=labels, axis=0)
    loss_2 = cross_entropy(input=logits.transpose(0,1), target=labels, axis=1) 
    loss = (loss_1 + loss_2)/2
    return loss