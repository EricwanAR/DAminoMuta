import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from math import cos, pi, sin
import math
import numpy as np
from scipy.special import lambertw



def mixup_criterion(criterion, pred, y_a, y_b, lam, pow=2):
    y = lam ** pow * y_a + (1 - lam) ** pow * y_b
    return criterion(pred, y)


def mixup_data(v, q, a):
    '''Returns mixed inputs, pairs of targets, and lambda without organ constraint'''
    lam = np.random.beta(1, 1)

    batch_size = v.shape[0]
    index = torch.randperm(batch_size)

    mixed_v = lam * v + (1 - lam) * v[index, :]
    mixed_q = lam * q + (1 - lam) * q[index, :]

    a_1, a_2 = a, a[index]
    return mixed_v, mixed_q, a_1, a_2, lam


def linear(epoch, nepoch):
    return 1 - epoch / nepoch


def convex(epoch, nepoch):
    return epoch / (2 - nepoch)


def concave(epoch, nepoch):
    return 1 - sin((epoch / nepoch) * (pi / 2))


def composite(epoch, nepoch):
    return 0.5 * cos((epoch / nepoch) * pi) + 0.5


class LogCoshLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))+F.mse_loss(y_t, y_prime_t)
    

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_t, weights=None):
        loss = (y - y_t) ** 2
        if weights is not None:
            loss *= weights.expand_as(loss)
        return torch.mean(loss)


class MLCE(nn.Module):
    def __init__(self):
        super(MLCE, self).__init__()

    def _mlcce(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        loss = torch.mean(neg_loss + pos_loss)
        return loss

    def __call__(self, y_pred, y_true):
        return self._mlcce(y_pred, y_true)


class SuperLoss(nn.Module):
    def __init__(self, C=10, lam=1, batch_size=256):
        super(SuperLoss, self).__init__()
        self.tau = math.log(C)
        self.lam = lam  # set to 1 for CIFAR10 and 0.25 for CIFAR100
        self.batch_size = batch_size

    def forward(self, logits, targets):
        l_i = F.mse_loss(logits, targets, reduction='none').detach()
        sigma = self.sigma(l_i)
        loss = (F.mse_loss(logits, targets, reduction='none') - self.tau) * sigma + self.lam * (
                torch.log(sigma) ** 2)
        loss = loss.sum() / self.batch_size
        return loss

    def sigma(self, l_i):
        x = torch.ones_like(l_i) * (-2 / math.exp(1.))
        y = 0.5 * torch.max(x, (l_i - self.tau) / self.lam)
        y = y.cpu().numpy()
        sigma = np.exp(-lambertw(y))
        sigma = sigma.real.astype(np.float32)
        sigma = torch.from_numpy(sigma).to(l_i.device)
        return sigma


def unbiased_curriculum_loss(out, data, args, epoch, epochs, scheduler='linear'):
    losses = []
    scheduler = linear if scheduler == 'linear' else concave

    # calculate difficulty measurement function
    adjusted_losses = []
    for idx in range(out.shape[0]):
        ground_truth = max(1, abs(data[idx].item()))
        loss = F.mse_loss(out[idx], data[idx])
        losses.append(loss)
        adjusted_losses.append(loss.item() / ground_truth)

    mean_loss, std_loss = np.mean(adjusted_losses), np.std(adjusted_losses)

    # re-weight losses
    total_loss = 0
    for i, loss in enumerate(losses):
        if adjusted_losses[i] > mean_loss + 1 * std_loss:
            schedule_factor = scheduler(epoch, args.epochs)
            total_loss += schedule_factor * loss
        else:
            total_loss += loss

    return total_loss


class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma=1.0):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def bmc_loss(self, pred, target, noise_var):
        """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
        Args:
        pred: A float tensor of size [batch, 1].
        target: A float tensor of size [batch, 1].
        noise_var: A float number or tensor.
        Returns:
        loss: A float tensor. Balanced MSE Loss.
        """
        if len(pred.shape) == 1:
            pred = pred.unsqueeze(1)
        if len(target.shape) == 1:
            target = target.unsqueeze(1)
        logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0], device=pred.device))     # contrastive-like loss
        loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

        return loss

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return self.bmc_loss(pred, target, noise_var)
