import torch
from gpytorch.kernels import RBFKernel

def mmd_loss(embedding, auxiliary_labels, weights_pos, weights_neg, sigma):
    """ Computes mmd loss, weighted or unweighted """
    if weights_pos is None:
        return mmd_loss_unweighted(embedding, auxiliary_labels, sigma)
    return mmd_loss_weighted(embedding, auxiliary_labels,
                             weights_pos, weights_neg, sigma)


def mmd_loss_unweighted(embedding, auxiliary_labels, sigma):
    kernel = RBFKernel(ard_num_dims=embedding.size(1), lengthscale=sigma)
    kernel_mat = kernel(embedding, embedding)

    if len(auxiliary_labels.shape) == 1:
        auxiliary_labels = auxiliary_labels.unsqueeze(-1)

    pos_mask = torch.mm(auxiliary_labels, auxiliary_labels.t())
    neg_mask = torch.mm(1.0 - auxiliary_labels,
                        (1.0 - auxiliary_labels).t())
    pos_neg_mask = torch.mm(auxiliary_labels,
                            (1.0 - auxiliary_labels).t())

    pos_kernel_mean = torch.div(
        torch.sum(pos_mask * kernel_mat), torch.sum(pos_mask))
    neg_kernel_mean = torch.div(
        torch.sum(neg_mask * kernel_mat), torch.sum(neg_mask))
    pos_neg_kernel_mean = torch.div(
        torch.sum(pos_neg_mask * kernel_mat), torch.sum(pos_neg_mask))

    mmd_val = pos_kernel_mean + neg_kernel_mean - 2 * pos_neg_kernel_mean
    mmd_val = torch.clamp_min(mmd_val, 0.0)

    return mmd_val, pos_kernel_mean, neg_kernel_mean, pos_neg_kernel_mean, pos_neg_kernel_mean


def mmd_loss_weighted(embedding, auxiliary_labels, weights_pos, weights_neg, sigma):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Decide which device we want to run on

    # Move all tensors to the chosen device
    embedding = embedding.to(device)
    auxiliary_labels = auxiliary_labels.to(device)
    weights_pos = weights_pos.to(device)
    weights_neg = weights_neg.to(device)
    #print(embedding.shape)

    # If the weights are 1D tensors, add an additional dimension
    if len(weights_pos.shape) == 1:
        weights_pos = weights_pos.unsqueeze(0)
    if len(weights_neg.shape) == 1:
        weights_neg = weights_neg.unsqueeze(0)

    kernel = RBFKernel(ard_num_dims=embedding.size(1), lengthscale=sigma).to(device)
    kernel_mat = kernel(embedding, embedding)

    #print(kernel_mat.shape)

    if len(auxiliary_labels.shape) == 1:
        auxiliary_labels = auxiliary_labels.unsqueeze(-1)
        auxiliary_labels = auxiliary_labels.unsqueeze(-1)

    pos_mask = torch.bmm(auxiliary_labels, auxiliary_labels.transpose(-2, -1))
    neg_mask = torch.bmm(1.0 - auxiliary_labels, (1.0 - auxiliary_labels).transpose(-2, -1))


    pos_neg_mask = torch.bmm(auxiliary_labels, (1.0 - auxiliary_labels).transpose(-2, -1))
    neg_pos_mask = torch.bmm((1.0 - auxiliary_labels), auxiliary_labels.transpose(-2, -1))

    pos_kernel_mean = kernel_mat * pos_mask
    pos_kernel_mean = pos_kernel_mean * weights_pos.t()

    pos_kernel_mean = torch.div(pos_kernel_mean.sum(dim=1), weights_pos.sum())
    pos_kernel_mean = torch.div((pos_kernel_mean * weights_pos.squeeze()).sum(), weights_pos.sum())

    neg_kernel_mean = kernel_mat * neg_mask
    neg_kernel_mean = neg_kernel_mean * weights_neg.t()

    neg_kernel_mean = torch.div(neg_kernel_mean.sum(dim=1), weights_neg.sum())
    neg_kernel_mean = torch.div((neg_kernel_mean * weights_neg.squeeze()).sum(), weights_neg.sum())

    neg_pos_kernel_mean = kernel_mat * neg_pos_mask
    neg_pos_kernel_mean = neg_pos_kernel_mean * weights_pos.t()

    neg_pos_kernel_mean = torch.div(neg_pos_kernel_mean.sum(dim=1), weights_pos.sum())
    neg_pos_kernel_mean = torch.div((neg_pos_kernel_mean * weights_neg.squeeze()).sum(), weights_neg.sum())

    pos_neg_kernel_mean = kernel_mat * pos_neg_mask
    pos_neg_kernel_mean = pos_neg_kernel_mean * weights_neg.t()

    pos_neg_kernel_mean = torch.div(pos_neg_kernel_mean.sum(dim=1), weights_neg.sum())
    pos_neg_kernel_mean = torch.div((pos_neg_kernel_mean * weights_pos.squeeze()).sum(), weights_pos.sum())

    mmd_val = pos_kernel_mean + neg_kernel_mean - (pos_neg_kernel_mean + neg_pos_kernel_mean)
    mmd_val = torch.clamp_min(mmd_val, 0.0)

    return mmd_val, pos_kernel_mean, neg_kernel_mean, pos_neg_kernel_mean, pos_neg_kernel_mean
