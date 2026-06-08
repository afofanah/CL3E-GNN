import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CurriculumLoss(nn.Module):
    """
    Three-phase curriculum loss (Eq.25-29).
    Label smoothing kept LOW (0.05) so floor stays near 0.43 not 0.86.
    """
    def __init__(self, num_classes, curriculum_type='performance',
                 initial_weight=0.3, final_weight=1.0,
                 total_epochs=1000, warmup_epochs=100, smoothing=0.05):
        super().__init__()
        self.num_classes     = num_classes
        self.curriculum_type = curriculum_type
        self.initial_weight  = initial_weight
        self.final_weight    = final_weight
        self.total_epochs    = total_epochs
        self.warmup_epochs   = warmup_epochs
        self.smoothing       = smoothing
        self.register_buffer('class_weights',  torch.ones(num_classes))
        self.register_buffer('class_accuracy', torch.zeros(num_classes))
        self.register_buffer('class_samples',  torch.zeros(num_classes))
        self.current_epoch = 0

    def update_epoch(self, epoch):
        self.current_epoch = epoch
        if self.curriculum_type == 'time':
            p = min(1.0, epoch / max(self.total_epochs, 1))
            self.class_weights.fill_(
                self.initial_weight + (self.final_weight - self.initial_weight) * p)

    def update_class_performance(self, outputs, targets):
        preds = outputs.argmax(1)
        for c in range(self.num_classes):
            mask = (targets == c)
            if mask.sum() > 0:
                acc = (preds[mask] == targets[mask]).float().mean().item()
                self.class_accuracy[c] = (
                    0.9 * self.class_accuracy[c] + 0.1 * acc
                    if self.class_samples[c] > 0 else acc)
                self.class_samples[c] += mask.sum().item()
        if (self.curriculum_type == 'performance'
                and self.current_epoch >= self.warmup_epochs):
            diff = (1.0 - self.class_accuracy).clamp(min=0)
            if diff.max() > 0:
                diff = diff / diff.max()
            p    = min(1.0, self.current_epoch / max(self.total_epochs, 1))
            base = self.initial_weight + (self.final_weight - self.initial_weight) * p
            self.class_weights = (base + (1.0 - base) * diff).to(self.class_weights.device)

    def forward(self, outputs, targets):
        if self.smoothing > 0:
            nc   = outputs.size(1)
            s    = self.smoothing
            oh   = F.one_hot(targets, nc).float()
            soft = (1 - s) * oh + s / nc
            loss = -(soft * F.log_softmax(outputs, 1)).sum(1)
        else:
            loss = F.cross_entropy(outputs, targets, reduction='none')
        return (loss * self.class_weights.to(outputs.device)[targets]).mean()


class EntropyRegularizedLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.1, beta=0.02,
                 dynamic_weighting=True, confidence_threshold=0.9):
        super().__init__()
        self.num_classes          = num_classes
        self.alpha                = alpha
        self.beta                 = beta
        self.dynamic_weighting    = dynamic_weighting
        self.confidence_threshold = confidence_threshold
        self.running_accuracy = 0.0
        self.running_entropy  = 0.0
        self.update_count     = 0

    def _entropy(self, probs):
        return -(probs * torch.log(probs + 1e-8)).sum(1) / (np.log(self.num_classes) + 1e-8)

    def update_weights(self, outputs, targets):
        if not self.dynamic_weighting:
            return
        probs = F.softmax(outputs.detach(), 1)
        acc   = (probs.argmax(1) == targets).float().mean().item()
        ent   = self._entropy(probs).mean().item()
        self.update_count += 1
        if self.update_count == 1:
            self.running_accuracy, self.running_entropy = acc, ent
        else:
            self.running_accuracy = 0.9 * self.running_accuracy + 0.1 * acc
            self.running_entropy  = 0.9 * self.running_entropy  + 0.1 * ent
        af         = 1.0 - min(1.0, self.running_accuracy / self.confidence_threshold)
        self.alpha = 0.01 + 0.09 * af
        self.beta  = 0.005 + 0.015 * (1.0 - self.running_entropy)

    def forward(self, outputs, targets):
        probs    = F.softmax(outputs, 1)
        ce       = F.cross_entropy(outputs, targets, reduction='none')
        ent      = self._entropy(probs)
        correct  = (probs.argmax(1) == targets).float()
        avg_p    = probs.mean(0)
        div      = F.kl_div(torch.log(avg_p + 1e-8),
                            torch.ones_like(avg_p) / self.num_classes,
                            reduction='sum')
        return (ce + self.alpha * correct * ent
                + self.beta * div / outputs.size(0)).mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, curriculum_weight=0.9, entropy_weight=0.1,
                 curriculum_args=None, entropy_args=None):
        super().__init__()
        self.curriculum_weight = curriculum_weight
        self.entropy_weight    = entropy_weight
        self.curriculum_loss   = CurriculumLoss(num_classes, **(curriculum_args or {}))
        self.entropy_loss      = EntropyRegularizedLoss(num_classes, **(entropy_args or {}))

    def update_epoch(self, epoch):
        self.curriculum_loss.update_epoch(epoch)

    def update_class_performance(self, outputs, targets):
        self.curriculum_loss.update_class_performance(outputs, targets)
        self.entropy_loss.update_weights(outputs, targets)

    def forward(self, outputs, targets):
        return (self.curriculum_weight * self.curriculum_loss(outputs, targets)
                + self.entropy_weight  * self.entropy_loss(outputs, targets))


def get_loss(loss_type, num_classes, curriculum_args=None, entropy_args=None):
    if loss_type == 'standard':
        # smoothing=0.05 → floor ~0.43 for 7 classes (vs 0.86 at 0.1)
        return nn.CrossEntropyLoss(label_smoothing=0.05)
    elif loss_type == 'curriculum':
        return CurriculumLoss(num_classes, **(curriculum_args or {}))
    elif loss_type == 'entropy':
        return EntropyRegularizedLoss(num_classes, **(entropy_args or {}))
    elif loss_type == 'combined':
        return CombinedLoss(num_classes,
                            curriculum_args=curriculum_args,
                            entropy_args=entropy_args)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")