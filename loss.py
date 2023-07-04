"""loss.py: losses used when training ATDA."""
import torch
import torch.nn as nn


def coral(source, target):
    """
    Coral loss
    """
    d = source.size(1)
    source_c = compute_covariance(source)
    target_c = compute_covariance(target)
    loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))
    loss = loss / (4 * d * d)
    return loss


def compute_covariance(input_data):
    """
    Compute Covariance matrix of the input data
    """
    n = input_data.size(0)  # batch_size

    # Check if using gpu or cpu
    if input_data.is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    id_row = torch.ones(n).resize(1, n).to(device=device)
    sum_column = torch.mm(id_row, input_data)
    mean_column = torch.div(sum_column, n)
    term_mul_2 = torch.mm(mean_column.t(), mean_column)
    d_t_d = torch.mm(input_data.t(), input_data)
    c = torch.add(d_t_d, (-1 * term_mul_2)) * 1 / (n - 1)
    return c


def mmd(source, target):
    """
    MMD Distance
    """
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def guassian_kernel(
        source,
        target,
        kernel_mul=2.0,
        kernel_num=5,
        fix_sigma=None):
    """
    Gaussian kernel to be used in the mmd distance
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


class MarginLoss(nn.Module):
    """Margin loss.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=10, use_gpu=True):
        super(MarginLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(
                torch.randn(
                    self.num_classes,
                    self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(
                torch.randn(
                    self.num_classes,
                    self.feat_dim))

    def forward(self, features, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        alpha = 0.1
        batch_size = features.shape[0]
        feat_dim = features.shape[1]
        num_classes = self.num_classes
        len_features = features.shape[1]
        centers_batch = self.centers[labels]
        center_dist = torch.abs(features - centers_batch).sum(dim=1)
        diff = centers_batch - features
        unique_label, unique_idx, unique_count = torch.unique(
            labels, return_inverse=True, return_counts=True)
        appear_time = unique_count[unique_idx].unsqueeze(
            1).expand(batch_size, feat_dim)

        diff = diff / (1 + appear_time).float()
        diff = alpha * diff

        feature_center_pair_dist = torch.abs(
            features.unsqueeze(1).expand(
                batch_size,
                num_classes,
                feat_dim) -
            self.centers.unsqueeze(0).expand(
                batch_size,
                num_classes,
                feat_dim)).sum(
            dim=2)
        feature_center_dist = center_dist.unsqueeze(1).expand(
            batch_size, num_classes) - feature_center_pair_dist

        classes = torch.arange(num_classes).long().cuda()
        feature_center_labels_equal = labels.unsqueeze(1).expand(
            batch_size, num_classes).eq(
            classes.expand(
                batch_size, num_classes))
        mask_feature_center = (~feature_center_labels_equal).float()
        margin_loss = (torch.nn.functional.softplus(
            feature_center_dist) * mask_feature_center).sum() / mask_feature_center.sum()
        return margin_loss
