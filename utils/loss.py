import torch


def percepLoss(pred, target, weight=1.0, loss_fn=None):

    N, T, C, H, W = pred.shape
    losses = loss_fn(pred.reshape(-1, C, H, W), target.reshape(-1, C, H, W))  
    loss_ts = losses.reshape(N, T).mean(dim=0)
    loss = torch.mean(loss_ts * weight)
    return loss, loss_ts


@torch.jit.script
def weightedL1(pred, target, weight=None):

    if weight is None:
        weight = 1.0
    diff = torch.abs(pred - target)
    sum_t = torch.mean(diff, dim=[0, 2, 3, 4])
    loss = torch.mean(sum_t * weight)
    return loss, sum_t


@torch.jit.script
def weightedL2(pred, target, weight=None):

    if weight is None:
        weight = 1.0
    diff = (pred - target) ** 2
    sum_t = torch.mean(diff, dim=[0, 2, 3, 4])
    loss = torch.mean(sum_t * weight)
    return loss, sum_t