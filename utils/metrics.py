import torch
import numpy as np
from medpy import metric
from hausdorff import hausdorff_distance

def dice_coef(y_true, y_pred, thr=0.5, epsilon=0.001):
    inter_map = y_true * y_pred
    inter = inter_map.sum()
    den = y_true.sum() + y_pred.sum()
    dice = ((2 * inter) / (den + epsilon)) if den > 0 else 0
    return dice

def eval(y_true, y_pred, thr=0.5,):
    num_classes = y_pred.shape[1]
    all_res = []
    for c in range(num_classes):
        res = []
        c_true = y_true.to(torch.float32).squeeze(0).cpu().detach().numpy()
        c_pred = (y_pred > thr).to(torch.float32).squeeze(0)[c].cpu().detach().numpy()
        if c_pred.sum() == 0:
            res = [0, 0, 0]
        else:
            # res.append(metric.binary.dc(c_pred, c_true))
            res.append(dice_coef(c_true, c_pred))
            res.append(metric.binary.jc(c_pred, c_true))
            res.append(hausdorff_distance(c_true, c_pred) * 0.95)
        all_res.append(res)
    all_res = np.mean(np.array(all_res), axis=0)
    return all_res


