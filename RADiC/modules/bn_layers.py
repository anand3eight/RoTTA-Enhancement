import torch
import torch.nn as nn
from copy import deepcopy


class MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm2d, momentum):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum

        if bn_layer.track_running_stats and bn_layer.running_var is not None and bn_layer.running_mean is not None:
            self.register_buffer("source_median", deepcopy(bn_layer.running_mean))  # Rename to median
            self.register_buffer("source_var", deepcopy(bn_layer.running_var))  # Will use MAD
            self.source_num = bn_layer.num_batches_tracked

        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)

        self.register_buffer("target_median", torch.zeros_like(self.source_median))
        self.register_buffer("target_var", torch.ones_like(self.source_var))
        self.eps = bn_layer.eps

    def forward(self, x):
        raise NotImplementedError


class RobustBN1d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_median = torch.median(x, dim=0, keepdim=False)[0]  # (C,)
            mad = torch.median(torch.abs(x - b_median[None, :]), dim=0, keepdim=False)[0]  # Median Absolute Deviation
            b_var = (mad + self.eps) ** 2  # Convert MAD to variance estimate

            median = (1 - self.momentum) * self.source_median + self.momentum * b_median
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var

            self.source_median, self.source_var = deepcopy(median.detach()), deepcopy(var.detach())
            median, var = median.view(1, -1), var.view(1, -1)
        else:
            median, var = self.source_median.view(1, -1), self.source_var.view(1, -1)

        x = (x - median) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1)
        bias = self.bias.view(1, -1)

        return x * weight + bias


class RobustBN2d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_median = torch.median(x, dim=0, keepdim=False)[0]  # Reduce batch dim
            b_median = torch.median(b_median, dim=-1, keepdim=False)[0]  # Reduce height
            b_median = torch.median(b_median, dim=-1, keepdim=False)[0] # Reduce width
            mad = torch.abs(x - b_median[None, :, None, None])  # Compute absolute deviation
            mad = torch.median(mad, dim=0, keepdim=False)[0]  # Reduce batch dim
            mad = torch.median(mad, dim=-1, keepdim=False)[0]  # Reduce height
            mad = torch.median(mad, dim=-1, keepdim=False)[0]  # Reduce width
            b_var = (mad + self.eps) ** 2  # Convert MAD to variance estimate

            median = (1 - self.momentum) * self.source_median + self.momentum * b_median
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var

            self.source_median, self.source_var = deepcopy(median.detach()), deepcopy(var.detach())
            median, var = median.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            median, var = self.source_median.view(1, -1, 1, 1), self.source_var.view(1, -1, 1, 1)

        x = (x - median) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias
