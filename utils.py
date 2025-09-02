import random
import torch
import numpy as np
import logging
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
from typing import Iterator, Iterable, Tuple, Any

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pretrain_model(net, weights):
    net_keys = list(net.state_dict().keys())
    weights_keys = list(weights.keys())
    # assert(len(net_keys) <= len(weights_keys))
    i = 0
    j = 0
    while i < len(net_keys) and j < len(weights_keys):
        name_i = net_keys[i]
        name_j = weights_keys[j]
        if net.state_dict()[name_i].shape == weights[name_j].shape:
            net.state_dict()[name_i].copy_(weights[name_j].cpu())
            i += 1
            j += 1
        else:
            i += 1
    # print i, len(net_keys), j, len(weights_keys)
    return net


def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def zip_restart_dataloader(iter_a: Iterable, dataloader) -> Iterator[Tuple[Any, Any]]:
    it_a = iter(iter_a)

    def new_it_b():
        # 每次需要时，重新创建一个 DataLoader 的迭代器（相当于新一轮“epoch”）
        return iter(dataloader)

    it_b = new_it_b()

    while True:
        try:
            a = next(it_a)
        except StopIteration:
            return

        try:
            b = next(it_b)
        except StopIteration:
            it_b = new_it_b()
            try:
                b = next(it_b)
            except StopIteration:
                # DataLoader 为空的情况
                raise ValueError("DataLoader 为空，无法配对")

        yield a, b


class FDS(nn.Module):

    def __init__(self, feature_dim, bucket_num=100, bucket_start=7, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=5, sigma=2, momentum=0.9):
        super(FDS, self).__init__()
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.kernel_window = self._get_kernel_window(kernel, ks, sigma)
        self.half_ks = (ks - 1) // 2
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth

        self.register_buffer('epoch', torch.zeros(1).fill_(start_update))
        self.register_buffer('running_mean', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('num_samples_tracked', torch.zeros(bucket_num - bucket_start))

    @staticmethod
    def _get_kernel_window(kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            base_kernel = np.array(base_kernel, dtype=np.float32)
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(
                gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks) / sum(triang(ks))
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / sum(
                map(laplace, np.arange(-half_ks, half_ks + 1)))

        logging.info(f'Using FDS: [{kernel.upper()}] ({ks}/{sigma})')
        return torch.tensor(kernel_window, dtype=torch.float32).cuda()

    def _get_bucket_idx(self, label):
        label = np.float32(label.cpu())
        return max(min(int(label * np.float32(10)), self.bucket_num - 1), self.bucket_start)

    def _update_last_epoch_stats(self):
        self.running_mean_last_epoch = self.running_mean
        self.running_var_last_epoch = self.running_var

        self.smoothed_mean_last_epoch = F.conv1d(
            input=F.pad(self.running_mean_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)
        self.smoothed_var_last_epoch = F.conv1d(
            input=F.pad(self.running_var_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)

        assert self.smoothed_mean_last_epoch.shape == self.running_mean_last_epoch.shape, \
            "Smoothed shape is not aligned with running shape!"

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1)
        self.num_samples_tracked.zero_()

    def update_last_epoch_stats(self, epoch):
        if epoch == self.epoch + 1:
            self.epoch += 1
            self._update_last_epoch_stats()
            logging.info(f"Updated smoothed statistics of last epoch on Epoch [{epoch}]!")

    def _running_stats_to_device(self, device):
        if device == 'cpu':
            self.num_samples_tracked = self.num_samples_tracked.cpu()
            self.running_mean = self.running_mean.cpu()
            self.running_var = self.running_var.cpu()
        else:
            self.num_samples_tracked = self.num_samples_tracked.cuda()
            self.running_mean = self.running_mean.cuda()
            self.running_var = self.running_var.cuda()

    def update_running_stats(self, features, labels, epoch):
        if epoch < self.epoch:
            return

        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(0), "Dimensions of features and labels are not aligned!"

        self._running_stats_to_device('cpu')

        labels = labels.unsqueeze(1).cpu()
        labels = labels.squeeze(1).view(-1)

        features = features.contiguous().view(-1, self.feature_dim)

        buckets = np.array([self._get_bucket_idx(label) for label in labels])
        for bucket in np.unique(buckets):
            curr_feats = features[torch.tensor((buckets == bucket).astype(bool))]
            curr_num_sample = curr_feats.size(0)
            curr_mean = torch.mean(curr_feats, 0)
            curr_var = torch.var(curr_feats, 0, unbiased=True if curr_feats.size(0) != 1 else False)

            self.num_samples_tracked[bucket - self.bucket_start] += curr_num_sample
            factor = self.momentum if self.momentum is not None else \
                (1 - curr_num_sample / float(self.num_samples_tracked[bucket - self.bucket_start]))
            factor = 0 if epoch == self.start_update else factor

            # print(curr_mean.is_cuda)
            self.running_mean[bucket - self.bucket_start] = \
                (1 - factor) * curr_mean + factor * self.running_mean[bucket - self.bucket_start]
            self.running_var[bucket - self.bucket_start] = \
                (1 - factor) * curr_var + factor * self.running_var[bucket - self.bucket_start]

        self._running_stats_to_device('cuda')
        logging.info(f"Updated running statistics with Epoch [{epoch}] features!")

    def smooth(self, features, labels, epoch):
        if epoch < self.start_smooth:
            return features

        labels = labels.unsqueeze(1)
        sp = labels.squeeze(1).shape

        labels = labels.squeeze(1).view(-1)
        features = features.contiguous().view(-1, self.feature_dim)

        buckets = torch.max(torch.stack([torch.min(torch.stack([torch.floor(labels * torch.tensor([10.]).cuda()).int(),
                                                                torch.zeros(labels.size(0)).fill_(
                                                                    self.bucket_num - 1).int().cuda()], 0), 0)[0],
                                         torch.zeros(labels.size(0)).fill_(self.bucket_start).int().cuda()], 0), 0)[0]
        for bucket in torch.unique(buckets):
            features[buckets.eq(bucket)] = calibrate_mean_var(
                features[buckets.eq(bucket)],
                self.running_mean_last_epoch[bucket.item() - self.bucket_start],
                self.running_var_last_epoch[bucket.item() - self.bucket_start],
                self.smoothed_mean_last_epoch[bucket.item() - self.bucket_start],
                self.smoothed_var_last_epoch[bucket.item() - self.bucket_start]
            )

        return features.view(*sp, self.feature_dim)
