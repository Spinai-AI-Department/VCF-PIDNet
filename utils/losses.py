import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_msssim import MS_SSIM
try:
    from PIDNet.utils.utils import Custom_loss
    from PIDNet.utils.criterion import BondaryLoss, CrossEntropy, OhemCrossEntropy, Focal
except:
    from .PIDNet.utils.utils import Custom_loss
    from .PIDNet.utils.criterion import BondaryLoss, CrossEntropy, OhemCrossEntropy, Focal
SMOOTH = 1.0e-8


def get_loss(loss_name, loss_type):

    if loss_name == "pidnet-custom":
        
        if loss_type == 'ce':
            sem_func = CrossEntropy()
        elif loss_type == 'fc':
            sem_func = Focal()
        else:
            sem_func = OhemCrossEntropy(ignore_label=255, thres= 0.9, min_kept=131072)
        bn_loss = BondaryLoss()

        loss = Custom_loss(sem_func, bn_loss)
        
        return loss
    elif loss_name=="hybrid":
        return HybridLoss()
    elif loss_name=="bce":
        return nn.BCELoss()
    elif loss_name == "ce":
        return nn.CrossEntropyLoss()
    elif loss_name=="dice":
        return DiceLoss()
    elif loss_name=="focal":
        return FocalLoss()
    # elif loss_name=="ms_ssim":
    #     return MS_SSIM(data_range=1, size_average=True, channel=12) # You should minus this from 1.
    else:
        raise RuntimeError("There is no loss named {}".format(loss_name))

class HybridLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.ms_ssim_loss = MS_SSIM(data_range=1, size_average=True, channel=12)

    def forward(self, pred, target):

        dl = self.dice_loss(pred, target)
        fl = self.focal_loss(pred, target)
        ml = 1 - self.ms_ssim_loss(pred, target)

        return dl + fl + ml

class DiceLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):



        numerator = 2 * (pred*target).sum(dim=(-2, -1))
        denominator = pred.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))

        return 1 - (numerator/(denominator+SMOOTH)).mean()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()


    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        

        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
