import cv2
import torch
import numpy as np
from medpy import metric
import torch.nn.functional as F
import matplotlib.pyplot as plt
from hausdorff import hausdorff_distance


def dice_coef(y_true, y_pred, thr=0.5, epsilon=0.001):
    if y_pred.shape[1] > 1:
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0)[1].cpu().detach().numpy()
    else:
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0).squeeze(0).cpu().detach().numpy()
        
    y_true = (y_true > thr).to(torch.float32).squeeze(0).cpu().detach().numpy()
    inter_map = y_true * y_pred
    inter = inter_map.sum()
    den = y_true.sum() + y_pred.sum()
    # dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    dice = ((2 * inter) / (den + epsilon)) if den > 0 else 0
    # return dice, y_true, y_pred
    return dice


def evaluate_95hd(y_true, y_pred, thr=0.5):
    y_true = y_true.to(torch.float32).squeeze(0).cpu().detach().numpy()
    if y_pred.shape[1] > 1:
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0)[1].cpu().detach().numpy()
    else:
        y_pred = (y_pred > thr).to(torch.float32).squeeze(0).squeeze(0).cpu().detach().numpy()
    
    hd = hausdorff_distance(y_true, y_pred)
    return hd * 0.95




def patients_to_slices(dataset, patiens_num):
    ref_dict = {}
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "PROMISE12" in dataset:
        ref_dict = {"3": 101, "7": 202, "9": 303,}
    elif "BUSI" in dataset:
        ref_dict = {"1": 5, "5": 26, "10": 52, "20": 104, "30": 155, "50": 260}
    elif "tumor" in dataset:
        ref_dict = {"10": 145, "20": 290, "30": 435, }
    elif "ISIC" in dataset:
        ref_dict = {"10": 207, "30": 622, }
    elif "thyroid" in dataset:
        ref_dict = {"10": 613, "30": 1841, }
    elif "BrainMRI" in dataset:
        ref_dict = {"10": 103, "30": 310, "50": 515,}
    elif "MRI_Hippocampus_Seg" in dataset:
        ref_dict = {"10": 282, "30": 846, }
    elif "BCSS" in dataset:
        ref_dict = {"5": 136, "10": 272, "30": 816, }
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()
