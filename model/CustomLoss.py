import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, y_pred, y_true):
        n = y_pred.size(0)
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        num_classes = 5
        # 分层逻辑，计算层级标签
        _, indices_pred = torch.sort(y_pred, descending=False)
        _, indices_true = torch.sort(y_true, descending=False)

        labels_pred = torch.zeros_like(y_pred, dtype=torch.long)
        labels_true = torch.zeros_like(y_true, dtype=torch.long)

        # 计算每层的基本大小和需要额外分配的数据点数
        base_size, extra = divmod(n, num_classes)

        start_idx = 0
        for i in range(num_classes):
            # 对于前extra层，每层多分配一个数据点
            size = base_size + (1 if i < extra else 0)
            end_idx = start_idx + size

            labels_pred[indices_pred[start_idx:end_idx]] = i
            labels_true[indices_true[start_idx:end_idx]] = i

            start_idx = end_idx

        # 计算交叉熵损失
        a,b = labels_pred.shape,labels_true.shape
        loss = F.cross_entropy(F.one_hot(labels_pred, num_classes=num_classes).float(), labels_true)
        return loss

class GruopAccuracy(nn.Module):
    def __init__(self):
        super(GruopAccuracy, self).__init__()

    def forward(self, y_pred, y_true):
        try:
            n = y_pred.size(0)
        except:
            n = y_pred.size
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        num_classes = 5
        # 分层逻辑，计算层级标签
        _, indices_pred = torch.sort(y_pred, descending=False)
        _, indices_true = torch.sort(y_true, descending=False)

        labels_pred = torch.zeros_like(y_pred, dtype=torch.long)
        labels_true = torch.zeros_like(y_true, dtype=torch.long)

        # 计算每层的基本大小和需要额外分配的数据点数
        base_size, extra = divmod(n, num_classes)

        start_idx = 0
        for i in range(num_classes):
            # 对于前extra层，每层多分配一个数据点
            size = base_size + (1 if i < extra else 0)
            end_idx = start_idx + size

            labels_pred[indices_pred[start_idx:end_idx]] = i
            labels_true[indices_true[start_idx:end_idx]] = i

            start_idx = end_idx

        # 计算交叉熵损失
        acc = torch.mean(((torch.abs(labels_pred - labels_true) <= 1).float()))
        loss = acc * (-1)
        return loss


class OrdinalRegressionLoss(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # 确保y_pred和y_true为浮点数，并且有相同的形状
        n = y_pred.size(0)
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)
        num_classes = 5
        # 分层逻辑，计算层级标签
        _, indices_pred = torch.sort(y_pred, descending=False)
        _, indices_true = torch.sort(y_true, descending=False)

        labels_pred = torch.zeros_like(y_pred, dtype=torch.long)
        labels_true = torch.zeros_like(y_true, dtype=torch.long)

        # 计算每层的基本大小和需要额外分配的数据点数
        base_size, extra = divmod(n, num_classes)

        start_idx = 0
        for i in range(num_classes):
            # 对于前extra层，每层多分配一个数据点
            size = base_size + (1 if i < extra else 0)
            end_idx = start_idx + size

            labels_pred[indices_pred[start_idx:end_idx]] = i
            labels_true[indices_true[start_idx:end_idx]] = i

            start_idx = end_idx
        labels_true = labels_true.float()
        labels_pred = labels_pred.float()

        # 计算预测和真实标签之间的绝对差异
        difference = torch.abs(labels_pred - labels_true)

        # 计算损失，对远离真实标签的预测给予更大的惩罚
        loss = torch.mean(difference ** 2)  # 使用平方差以放大较大差异的惩罚

        return loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # 确保y_pred和y_true为浮点数，并且有相同的形状
        y_pred = y_pred.float()
        y_true = y_true.float()

        # 计算预测值和真实值之间的均方差损失
        loss = torch.mean((y_pred - y_true) ** 2)

        return loss

from scipy.stats import spearmanr

class RankCorrelationLoss(nn.Module):
    def __init__(self):
        super(RankCorrelationLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # 计算秩相关系数
        correlation, _ = spearmanr(y_pred.detach().cpu().numpy(), y_true.detach().cpu().numpy())

        # 将秩相关系数转换为 PyTorch 张量并返回
        correlation_tensor = torch.tensor(correlation, dtype=torch.float32)

        # 将秩相关系数作为损失返回
        return -correlation_tensor  # 使用负值，因为优化器默认最小化损失


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))

class DistanceCorrelationLoss(nn.Module):
    def __init__(self):
        super(DistanceCorrelationLoss, self).__init__()

    def forward(self, x, y):
        if x.dim() == 1:
            x = x.view(-1, 1)
        if y.dim() == 1:
            y = y.view(-1, 1)

        # 计算距离矩阵
        dist_x = torch.cdist(x, x)
        dist_y = torch.cdist(y, y)

        # 计算平均距离并双中心化
        mean_x = dist_x.mean(0, keepdim=True)
        mean_y = dist_y.mean(0, keepdim=True)
        A = dist_x - mean_x - (dist_x - mean_x).mean(1, keepdim=True) + dist_x.mean()
        B = dist_y - mean_y - (dist_y - mean_y).mean(1, keepdim=True) + dist_y.mean()

        # 计算距离协方差和距离方差
        dcov2_xy = torch.mean(A * B)
        dcov2_xx = torch.mean(A * A)
        dcov2_yy = torch.mean(B * B)

        # 引入epsilon防止除以零
        epsilon = 1e-8
        dcor = torch.sqrt(dcov2_xy / (torch.sqrt(dcov2_xx * dcov2_yy) + epsilon))

        return 1 - dcor

import torch
import torch.nn as nn


class SmoothRankingLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SmoothRankingLoss, self).__init__()
        self.temperature = temperature

    def forward(self, y_pred, y_true):
        # 对y_pred和y_true使用softmax进行软排序
        sorted_pred_indices = torch.argsort(torch.argsort(y_pred))
        sorted_true_indices = torch.argsort(torch.argsort(y_true))

        # 计算softmax
        soft_pred = torch.softmax(y_pred / self.temperature, dim=0)
        soft_true = torch.softmax(y_true / self.temperature, dim=0)

        # 计算损失
        loss = torch.mean((soft_pred - soft_true) ** 2)
        return loss

