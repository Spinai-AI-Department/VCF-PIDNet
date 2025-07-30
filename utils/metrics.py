import torch
import torch.nn as nn
import torch.nn.functional as F

def get_metric(metric_name, device=None):

    if metric_name == "meanIoU":
        return meanIoU
    elif metric_name == "f1Score":
        return f1_score
    else:
        raise RuntimeError("There is no metric named {}".format(metric_name))



def meanIoU(y_pred, y_true):

    C = y_pred.size()[-3]

    arg_pred = torch.argmax(y_pred, dim=-3)

    y_pred = F.one_hot(arg_pred, num_classes=C).permute(0, 3, 1, 2).to(torch.int32)
    y_true = F.one_hot(y_true, num_classes=C).permute(0, 3, 1, 2).to(torch.int32)

    inter = y_true & y_pred
    union = y_true | y_pred
    mIoU = torch.sum(inter, dim=(-2, -1))/(torch.sum(union, dim=(-2, -1))+1e-8)


    return torch.mean(mIoU)

def f1_score(y_pred, y_true):

    C = y_pred.size()[-3]
    
    arg_pred = torch.argmax(y_pred, dim=-3)
    
    y_pred = F.one_hot(arg_pred, num_classes=C).permute(0, 3, 1, 2).to(torch.int32)
    y_true = F.one_hot(y_true, num_classes=C).permute(0, 3, 1, 2).to(torch.int32)
    
    TP_2 = (y_true & y_pred) * 2
    TP_2_FN_FP = y_true + y_pred

    f1 = torch.sum(TP_2, dim=(-2, -1)).to(torch.float32) / (torch.sum(TP_2_FN_FP, dim=(-2, -1)).to(torch.float32)+1e-8)

    return torch.mean(f1)