# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F
try:
    from configs import config
except:
    from ..configs import config

class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label
        )

    def _forward(self, score, target):

        loss = self.criterion(score, target)

        return loss

    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
        elif len(score) == 1:
            return sb_weights * self._forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")

class Focal(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_label=-1):
        super(Focal, self).__init__()
        # alpha: weight per class
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_label = ignore_label
    
    def _forward(self, score, targets):
        """
        inputs: [B, C, H, W] → raw logits
        targets: [B, H, W]   → class indices
        """
        if len(targets.shape) == 4:
            targets = targets.argmax(1)
        targets = targets.to(torch.long)
        # Mask for ignore_index
        valid_mask = (targets != self.ignore_label)  # [B, H, W]
        targets = targets.clone()
        targets[valid_mask == False] = 0  # temporarily fill with a valid class index
        # Compute log-softmax
        log_probs = F.log_softmax(score, dim=1)  # [B, C, H, W]
        probs = torch.exp(log_probs)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=score.shape[1])  # [B, H, W, C]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()       # [B, C, H, W]

        # Focal loss core
        pt = (probs * targets_one_hot).sum(dim=1)  # [B, H, W]
        focal_term = (1 - pt) ** self.gamma
        
        # === Class-wise alpha 적용 ===
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.device != score.device:
                self.alpha = self.alpha.to(score.device)
            alpha_t = (self.alpha[targets] * valid_mask).float()  # [B, H, W]
        else:
            alpha_t = self.alpha  # scalar alpha

        loss = -alpha_t * focal_term * (log_probs * targets_one_hot).sum(dim=1)  # [B, H, W]

        # Apply ignore mask
        loss = loss[valid_mask]

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # no reduction
    
    def forward(self, score, target):

        if config.MODEL.NUM_OUTPUTS == 1:
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            return sum([w * self._forward(x, target) for (w, x) in zip(balance_weights, score)])
        elif len(score) == 1:
            return sb_weights * self._forward(score[0], target)
        else:
            raise ValueError("lengths of prediction and target are not identical!")

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )

    def _ce_forward(self, score, target):


        loss = self.criterion(score, target)

        return loss

    def _ohem_forward(self, score, target, **kwargs):

        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, score, target):
        
        if not (isinstance(score, list) or isinstance(score, tuple)):
            score = [score]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        sb_weights = config.LOSS.SB_WEIGHTS
        if len(balance_weights) == len(score):
            functions = [self._ce_forward] * \
                (len(balance_weights) - 1) + [self._ohem_forward]
            return sum([
                w * func(x, target)
                for (w, x, func) in zip(balance_weights, score, functions)
            ])
        
        elif len(score) == 1:
            return sb_weights * self._ohem_forward(score[0], target)
        
        else:
            raise ValueError("lengths of prediction and target are not identical!")


def weighted_bce(bd_pre, target):
    n, c, h, w = bd_pre.size()
    log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)
    target_t = target.view(1, -1)

    pos_index = (target_t == 1)
    neg_index = (target_t == 0)

    weight = torch.zeros_like(log_p)
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')

    return loss


class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce = 20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce
        
    def forward(self, bd_pre, bd_gt):

        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss
        
        return loss
    
if __name__ == '__main__':
    a = torch.zeros(2,64,64)
    a[:,5,:] = 1
    pre = torch.randn(2,1,16,16)
    
    Loss_fc = BondaryLoss()
    loss = Loss_fc(pre, a.to(torch.uint8))

        
        
        


