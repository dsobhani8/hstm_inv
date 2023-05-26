import torch
from gpytorch.kernels import RBFKernel


def mmd_loss(embedding, auxiliary_labels, weights_pos=None, weights_neg=None, sigma=10):
    if weights_pos is None:
        return mmd_loss_unweighted(embedding, auxiliary_labels, sigma)
    return mmd_loss_weighted(embedding, auxiliary_labels, weights_pos, weights_neg, sigma)


def mmd_loss_unweighted(embedding, auxiliary_labels, sigma=10):
    kernel = RBFKernel(lengthscale=sigma)
    kernel_mat = kernel(embedding, embedding)

    if len(auxiliary_labels.shape) == 1:
        auxiliary_labels = auxiliary_labels.unsqueeze(-1)

    pos_mask = torch.mm(auxiliary_labels, auxiliary_labels.t())
    neg_mask = torch.mm(1.0 - auxiliary_labels, (1.0 - auxiliary_labels).t())
    pos_neg_mask = torch.mm(auxiliary_labels, (1.0 - auxiliary_labels).t())

    pos_kernel_mean = (pos_mask * kernel_mat).sum() / pos_mask.sum()
    neg_kernel_mean = (neg_mask * kernel_mat).sum() / neg_mask.sum()
    pos_neg_kernel_mean = (pos_neg_mask * kernel_mat).sum() / pos_neg_mask.sum()

    mmd_val = pos_kernel_mean + neg_kernel_mean - 2 * pos_neg_kernel_mean
    mmd_val = max(0.0, mmd_val)

    return mmd_val, pos_kernel_mean, neg_kernel_mean, pos_neg_kernel_mean, pos_neg_kernel_mean


def mmd_loss_weighted(embedding, auxiliary_labels, weights_pos, weights_neg, sigma=10):
	embedding = embedding.to(device)
	auxiliary_labels = auxiliary_labels.to(device)
	weights_pos = weights_pos.to(device)
	weights_neg = weights_neg.to(device)


kernel = RBFKernel(lengthscale=sigma)
    kernel_mat = kernel(embedding, embedding)

    if len(auxiliary_labels.shape) == 1:
        auxiliary_labels = auxiliary_labels.unsqueeze(-1)

    pos_mask = torch.mm(auxiliary_labels, auxiliary_labels.t())
    neg_mask = torch.mm(1.0 - auxiliary_labels, (1.0 - auxiliary_labels).t())
    pos_neg_mask = torch.mm(auxiliary_labels, (1.0 - auxiliary_labels).t())
    neg_pos_mask = torch.mm((1.0 - auxiliary_labels), auxiliary_labels.t())

    pos_kernel_mean = kernel_mat * pos_mask
    pos_kernel_mean = pos_kernel_mean * weights_pos.t()

    pos_kernel_mean = (pos_kernel_mean.sum(dim=1) / weights_pos.sum()).sum() / weights_pos.sum()

    neg_kernel_mean = kernel_mat * neg_mask
    neg_kernel_mean = neg_kernel_mean * weights_neg.t()

    neg_kernel_mean = (neg_kernel_mean.sum(dim=1) / weights_neg.sum()).sum() / weights_neg.sum()

    neg_pos_kernel_mean = kernel_mat * neg_pos_mask
    neg_pos_kernel_mean = neg_pos_kernel_mean * weights_pos.t()

    neg_pos_kernel_mean = (neg_pos_kernel_mean.sum(dim=1) / weights_pos.sum()).sum() / weights_neg.sum()

    pos_neg_kernel_mean = kernel_mat * pos_neg_mask
    pos_neg_kernel_mean = pos_neg_kernel_mean * weights_neg.t()

    pos_neg_kernel_mean = (pos_neg_kernel_mean.sum(dim=1) / weights_neg.sum()).sum() / weights_pos.sum()

    mmd_val = pos_kernel_mean + neg_kernel_mean - (pos_neg_kernel_mean + neg_pos_kernel_mean)
    mmd_val = max(0.0, mmd_val)

    return mmd_val, pos_kernel_mean, neg_kernel_mean, pos_neg_kernel_mean, pos_neg_kernel_mean
