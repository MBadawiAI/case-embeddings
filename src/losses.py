import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        log_pt = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]

        return loss.mean()


class GeneralizedCrossEntropyLoss(nn.Module):
    def __init__(self, q=0.7, class_weights=None, eps=1e-12):
        super().__init__()
        if q <= 0 or q > 1:
            raise ValueError("GCE requires 0 < q <= 1")
        self.q = q
        self.class_weights = class_weights
        self.eps = eps

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        pt = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        pt = pt.clamp_min(self.eps)

        loss = (1 - pt.pow(self.q)) / self.q

        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]

        return loss.mean()


def parse_loss_function(loss_cfg, device):
    name = loss_cfg.name

    class_weights = None
    if "class_weights" in loss_cfg and loss_cfg.class_weights is not None:
        class_weights = torch.tensor(
            loss_cfg.class_weights,
            dtype=torch.float,
            device=device,
        )

    if name == "cross_entropy":
        return nn.CrossEntropyLoss()

    if name == "weighted_cross_entropy":
        if class_weights is None:
            raise ValueError("weighted_cross_entropy requires loss.class_weights")
        return nn.CrossEntropyLoss(weight=class_weights)

    if name == "focal_loss":
        gamma = loss_cfg.gamma if "gamma" in loss_cfg else 2.0
        return FocalLoss(gamma=gamma, class_weights=class_weights)

    if name == "label_smoothing_cross_entropy":
        label_smoothing = (
            loss_cfg.label_smoothing if "label_smoothing" in loss_cfg else 0.0
        )
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    if name == "generalized_cross_entropy":
        q = loss_cfg.q if "q" in loss_cfg else 0.7
        return GeneralizedCrossEntropyLoss(q=q, class_weights=class_weights)

    raise ValueError(f"Unknown loss function: {name}")