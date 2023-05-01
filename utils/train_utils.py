import torch
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits

# adapted from https://cv-tricks.com/how-to/understanding-clip-by-openai/
# y_pred: (batch_size, embedding_size)
# y_gt: (batch_size, embedding_size)
def contrastive_loss(y_pred, y_gt, logit_scale):
    # y_pred = y_pred / torch.norm(y_pred, dim=1).unsqueeze(-1)
    # y_gt = y_gt / torch.norm(y_gt, dim=1).unsqueeze(-1)
    logits = logit_scale*torch.matmul(y_pred, y_gt.transpose(0,1))
    labels = torch.arange(y_pred.shape[0])
    if y_pred.is_cuda:
        labels = labels.to(y_pred.get_device())
    loss_1 = cross_entropy(input=logits, target=labels)
    loss_2 = cross_entropy(input=logits.transpose(0,1), target=labels) 
    # TODO: divide loss by batch size?
    loss = (loss_1 + loss_2)/2
    return loss

# y_pred: (batch_size, embedding_size)
# y_adv: (batch_size, embedding_size)
# y_gt: (batch_size, embedding_size)
def contrastive_adv_loss(y_pred, y_adv, y_gt, logit_scale):
    # y_pred = y_pred / torch.norm(y_pred, dim=1).unsqueeze(-1)
    # y_gt = y_gt / torch.norm(y_gt, dim=1).unsqueeze(-1)
    logits = logit_scale*torch.matmul(y_pred, y_gt.transpose(0,1))
    labels = torch.arange(y_pred.shape[0])
    if y_pred.is_cuda:
        labels = labels.to(y_pred.get_device())
    loss_1 = cross_entropy(input=logits, target=labels)
    adv_logits = torch.sum(y_adv * y_gt, dim=1, keepdim=True)
    logits_2 = torch.cat([logits.transpose(0,1), adv_logits], dim=1)
    loss_2 = cross_entropy(input=logits_2, target=labels) 
    # TODO: divide loss by batch size?
    loss = (loss_1 + loss_2)/2
    return loss

def binary_adv_crossentropy_loss(y_pred, y_adv, y_gt, logit_scale):
    logits_pred = logit_scale * (y_pred * y_gt).sum(dim=1)
    logits_adv = logit_scale * (y_adv * y_gt).sum(dim=1)
    logits = torch.stack([logits_pred, logits_adv], dim=1)
    target = torch.zeros(len(y_pred), dtype=torch.int64)
    if y_pred.is_cuda:
        target = target.to(y_pred.get_device())
    loss = cross_entropy(input=logits, target=target)
    return loss

def adversarial_relation_loss(y_pred_reliable, y_pred_adversarial, y_gt, logit_scale):
    # y_pred_reliable: (batch_size, embedding_size)
    # y_pred_adversarial: (batch_size, embedding_size)
    # y_gt: (batch_size, embedding_size)
    # calculate distance between y_reliable, y_adversarial and y_gt
    logits_reliable = torch.bmm(y_pred_reliable.unsqueeze(1), y_gt.unsqueeze(2))
    logits_adversarial = torch.bmm(y_pred_adversarial.unsqueeze(1), y_gt.unsqueeze(2))
    logits = torch.cat([logits_reliable, logits_adversarial], dim=1)
    loss = cross_entropy(input=logits, target=torch.zeros(y_pred_reliable.shape[0],dtype=torch.long))
    return loss
