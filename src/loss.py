import torch
import torch.nn as nn
import torch.nn.functional as F
from lovasz_losses import lovasz_softmax


class PairwiseMarginLoss(nn.Module):
    def __init__(self, gamma=0.9):
        super(PairwiseMarginLoss, self).__init__()
        self.gamma = gamma

    def forward(self, similarity, y_true):
        # gamma_n = 1 - self.gamma
        loss_positive = torch.relu(2 * (self.gamma - similarity)) * y_true
        # loss_negative = torch.relu(similarity - gamma_n) * (1 - y_true)
        loss_negative = torch.relu(similarity) * (1 - y_true)
        return (loss_positive + loss_negative).mean()


class DynamicBoundaryLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(DynamicBoundaryLoss, self).__init__()
        self.margin = margin

    def forward(self, similarity, y_true):
        with torch.no_grad():
            pos_sim = similarity[y_true == 1]
            neg_sim = similarity[y_true == 0]
            gamma_p = pos_sim.mean().item() if len(pos_sim) > 0 else 0.9
            gamma_n = neg_sim.mean().item() if len(neg_sim) > 0 else 0.1
            gamma = gamma_p - self.margin
            gamma_neg = gamma_n + self.margin
        loss_pos = F.softplus(5 * (gamma - similarity)) * y_true
        loss_neg = F.softplus((similarity - gamma_neg)) * (1 - y_true)
        return (loss_pos + loss_neg).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        eps = 1e-7
        probs = torch.softmax(logits, dim=1)[:, 1]
        loss_1 = -1 * self.alpha * torch.pow((1 - probs), self.gamma) * torch.log(probs + eps) * labels
        loss_0 = -1 * (1 - self.alpha) * torch.pow(probs, self.gamma) * torch.log(1 - probs + eps) * (1 - labels)
        loss = loss_0 + loss_1
        return torch.mean(loss)


class MultiLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.cross_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, s, logits, label):
        return self.alpha * self.cross_loss(logits, label) + (1 - self.alpha) * self.mse_loss(s, label.float())


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, label):
        probs = torch.softmax(logits, dim=1)
        one_hot_labels = torch.nn.functional.one_hot(label, num_classes=2).float()
        return nn.MSELoss()(probs, one_hot_labels)


class NLLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, label):
        log_probs = F.log_softmax(logits, dim=1)
        return nn.NLLLoss()(log_probs, label.long())


class KLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, label):
        log_probs = F.log_softmax(logits, dim=1)
        one_hot_labels = F.one_hot(label, num_classes=2).float()
        return nn.KLDivLoss(reduction="batchmean")(log_probs, one_hot_labels)


class BCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, label):
        return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(logits[:, 1], label.float())


class SoftMarginLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, label):
        target = label.float() * 2 - 1
        return nn.SoftMarginLoss()(logits[:, 1], target)


class CosineEmbeddingLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, similarity, label):
        pos_mask = label == 1
        neg_mask = label == 0
        pos_loss = 1 - similarity[pos_mask]
        neg_loss = torch.clamp(similarity[neg_mask] - self.margin, min=0)
        loss = torch.cat([pos_loss, neg_loss]).mean()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        probs = F.softmax(logits, dim=1)
        probs_pos = probs[:, 1]
        targets_pos = targets_one_hot[:, 1]
        probs_flat = probs_pos.view(-1)
        targets_flat = targets_pos.view(-1)

        intersection = torch.sum(probs_flat * targets_flat)
        union = torch.sum(probs_flat) + torch.sum(targets_flat)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice_score
        return loss


class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()

    def forward(self, logits, labels):
        probs = F.log_softmax(logits, dim=1).unsqueeze(-1).unsqueeze(-1)
        loss = lovasz_softmax(probs, labels)
        return loss


class OhemCELoss(nn.Module):
    def __init__(self, thresh):
        super().__init__()
        self.thresh = -torch.log(torch.tensor(thresh))
        self.criteria = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        n_min = labels.numel() // 16
        loss = self.criteria(logits, labels)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return loss_hard.mean()


class SELoss(nn.Module):
    def __init__(self, num_classes=2, init_lambda=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.global_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.lambda_param = nn.Parameter(torch.tensor(init_lambda))
        self.lambda_activation = nn.Sigmoid()

    def forward(self, logits, labels):
        loss_ce = self.ce_loss(logits, labels)
        global_feat = logits.mean(dim=0)
        global_pred = self.global_fc(global_feat)
        loss_global = F.binary_cross_entropy_with_logits(
            global_pred,
            F.one_hot(labels, num_classes=2).float().mean(dim=0)
        )
        adaptive_lambda = self.lambda_activation(self.lambda_param)
        total_loss = (1 - adaptive_lambda) * loss_ce.mean() + adaptive_lambda * loss_global
        return total_loss


class GHMLoss(nn.Module):
    def __init__(self, bins=10, momentum=0.75):
        super().__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = torch.arange(bins + 1).float() / bins
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.register_buffer('acc_sum', torch.zeros(bins))

    def forward(self, logits, labels):
        probs = F.softmax(logits, dim=1)
        correct_probs = probs[range(len(labels)), labels]
        g = (1 - correct_probs).detach()
        weights = torch.zeros_like(g)
        valid = (labels >= 0)
        total_valid = valid.sum().item()
        bin_indices = torch.bucketize(g, self.edges.to(g.device))
        for i in range(self.bins):
            mask = (bin_indices == i) & valid
            num_in_bin = mask.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = self.momentum * self.acc_sum[i] + (1 - self.momentum) * num_in_bin
                    weights[mask] = total_valid / self.acc_sum[i]
                else:
                    weights[mask] = total_valid / num_in_bin
        weights = weights / weights.sum() * total_valid
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        return (ce_loss * weights).mean()