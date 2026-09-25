import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss
from utils.utils import dice_coef, evaluate_95hd, generate_mask
import numpy as np
from model.model_Jump import Semi_GMS
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler

from skimage.measure import label

def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    return torch.Tensor(batch_list)

def get_cut_mask(out, thres=0.5, nms=1, temperature=0.1):
    # probs = torch.nn.Sigmoid()(out)
    probs = out ** (1 / temperature)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 0, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks.cuda()


class PolyWarmRestartScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, base_lr, max_iters, power=0.9, warm_restart_iters=5000, last_epoch=-1):
        self.base_lr = base_lr
        self.max_iters = max_iters
        self.power = power
        self.warm_restart_iters = warm_restart_iters
        self.current_cycle_start = 0
        super(PolyWarmRestartScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch - self.current_cycle_start
        if t >= self.warm_restart_iters:
            self.current_cycle_start = self.last_epoch
            t = 0
        factor = (1 - t / self.warm_restart_iters) ** self.power
        return [self.base_lr * factor for _ in self.base_lrs]
    
class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, warmup_iters, eta_min=0, last_epoch=-1):
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.eta_min = eta_min
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch_tensor = torch.tensor(self.last_epoch, dtype=torch.float32)
        if self.last_epoch < self.warmup_iters:
            return [base_lr * (epoch_tensor / self.warmup_iters) for base_lr in self.base_lrs]
        else:
            cosine_lr = self.eta_min + (self.base_lrs[0] - self.eta_min) * 0.5 * (1 + torch.cos(torch.pi * (epoch_tensor - self.warmup_iters) / (self.max_iters - self.warmup_iters)))
            return [cosine_lr for _ in self.base_lrs]
    

def add_gaussian_noise(img, mean=0., std=0.1):
    noise = torch.randn_like(img) * std + mean
    noisy_img = img + noise
    noisy_img = torch.clamp(noisy_img, 0., 1.) 
    return noisy_img


class Trainer(nn.Module):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.mse_loss  = torch.nn.MSELoss(reduction='mean')
 
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(args.num_classes)
        
        self.model = Semi_GMS(args=self.args, in_chns=self.args.in_channels, class_num=self.args.num_classes, stage=self.args.stage).cuda().train()
        # self.model = nn.DataParallel(self.model, device_ids=[0, 1])

        
        if self.args.stage == "1":
            if isinstance(self.model, torch.nn.DataParallel):
                self.optimizer_1 = AdamW(self.model.module.mapping_model.parameters(), lr=0.002)
                self.optimizer_2 = torch.optim.SGD(self.model.module.resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
            else:
                self.optimizer_1 = AdamW(self.model.mapping_model.parameters(), lr=0.002)
                self.optimizer_2 = torch.optim.SGD(self.model.resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

            self.scheduler_1 = PolyWarmRestartScheduler(self.optimizer_1,
                                                base_lr=0.002, 
                                                max_iters=args.map_max_iterations, 
                                                power=0.9, 
                                                warm_restart_iters=args.map_warmup_iterations 
                                                )

            self.scheduler_2 = PolyWarmRestartScheduler(self.optimizer_2,
                                                base_lr=0.01, 
                                                max_iters=args.map_max_iterations, 
                                                power=0.9, 
                                                warm_restart_iters=args.map_warmup_iterations 
                                                )
            self.best_performance = 100.0

        else:
            if isinstance(self.model, torch.nn.DataParallel):
                self.optimizer = torch.optim.SGD(self.model.module.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
            self.scheduler = PolyWarmRestartScheduler(self.optimizer,
                                                base_lr=0.01,
                                                max_iters=args.seg_max_iterations, 
                                                power=0.9, 
                                                warm_restart_iters=args.seg_warmup_iterations 
                                                )
            self.best_performance = 0.0


    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return self.args.consistency * self.sigmoid_rampup(epoch, self.args.consistency_rampup)


    def get_entropy_map(self, p):
        p = torch.sigmoid(p)
        ent_map = -1 * torch.sum(p * torch.log(p + 1e-6), dim=1, keepdim=True)
        return ent_map

    
    def seg_loss(self, pred, mask):
        loss = self.dice_loss(pred, mask.unsqueeze(1))
        return loss
        

    def train(self, volume_batch, label_batch, logger, iter_num):
        
        if volume_batch.shape[1] == 1:
            volume_batch = volume_batch.repeat(1, 3, 1, 1)
        
        # [-1, 1]
        img_rgb = 2. * volume_batch - 1.
        seg_rgb = label_batch[:self.args.labeled_bs] / (self.args.num_classes - 1)
        # seg_rgb = 2. * label_batch[:self.args.labeled_bs] - 1.
        
        if self.args.stage == "1":
            vae_latent_mean, cnn_latent_mean, seg_latent_mean = self.model(img_rgb, seg_rgb, is_train=True)
            # vae_latent_mean, seg_latent_mean, vae_pred = self.model(img_rgb, seg_rgb, is_train=True)
            vae_sup_loss = F.mse_loss(vae_latent_mean[:self.args.labeled_bs], seg_latent_mean) 
            cnn_sup_loss = F.mse_loss(cnn_latent_mean[:self.args.labeled_bs], seg_latent_mean)
            vae_unsup_loss = F.mse_loss(vae_latent_mean[self.args.labeled_bs:], cnn_latent_mean[self.args.labeled_bs:].clone().detach())
            cnn_unsup_loss = F.mse_loss(cnn_latent_mean[self.args.labeled_bs:], vae_latent_mean[self.args.labeled_bs:].clone().detach())
            
            consistency_weight = self.get_current_consistency_weight(iter_num // int(self.args.map_max_iterations / self.args.consistency_rampup))
            vae_loss = vae_sup_loss + consistency_weight * vae_unsup_loss
            cnn_loss = cnn_sup_loss + consistency_weight * cnn_unsup_loss
            # vae_loss = vae_sup_loss
            # cnn_loss = cnn_sup_loss
            self.optimizer_1.zero_grad()
            self.optimizer_2.zero_grad()
            vae_loss.backward()
            cnn_loss.backward()
            self.optimizer_1.step()
            self.optimizer_2.step()
            self.scheduler_1.step()
            self.scheduler_2.step()
            
            logger.info('iteration %d : '
                        '  vae_loss : %f'
                        '  cnn_loss : %f'
                        '  vae_lr : %6f'
                        '  cnn_lr : %6f'

                        % (iter_num, 
                            vae_loss, 
                            cnn_loss, 
                            self.optimizer_1.param_groups[0]['lr'],
                            self.optimizer_2.param_groups[0]['lr']

                            ))
            
        else:
            # pred sup loss
            vae_pred, cnn_pred = self.model(img_rgb, None, is_train=True)
            vae_pred_sup_loss = self.seg_loss(vae_pred[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])
            cnn_pred_sup_loss = self.seg_loss(cnn_pred[:self.args.labeled_bs], label_batch[:self.args.labeled_bs])
            sup_loss = vae_pred_sup_loss + cnn_pred_sup_loss
            
            vae_pseudo_map = get_cut_mask(vae_pred[self.args.labeled_bs:])
            cnn_pseudo_map = get_cut_mask(cnn_pred[self.args.labeled_bs:])
            
            vae_pred_unsup_loss = self.seg_loss(vae_pred[self.args.labeled_bs:], cnn_pseudo_map)
            cnn_pred_unsup_loss = self.seg_loss(cnn_pred[self.args.labeled_bs:], vae_pseudo_map)
            unsup_loss = vae_pred_unsup_loss + cnn_pred_unsup_loss

            consistency_weight = self.get_current_consistency_weight(iter_num // int(self.args.seg_max_iterations / self.args.consistency_rampup))
            
            loss = sup_loss + consistency_weight * unsup_loss
            
            if iter_num > self.args.seg_warmup_iterations:
                # ==================== mix ============================================
                img_mask, loss_mask = generate_mask(volume_batch)
                img_rgb_mix_lab = img_rgb[:self.args.labeled_bs] * img_mask + img_rgb[self.args.labeled_bs:] * (1 - img_mask)
                img_rgb_mix_unlab = img_rgb[:self.args.labeled_bs] * (1 - img_mask) + img_rgb[self.args.labeled_bs:] * img_mask
                img_rgb_mix = torch.concat([img_rgb_mix_lab, img_rgb_mix_unlab], dim=0)

                vae_mix_lab_label = label_batch[:self.args.labeled_bs] * img_mask + vae_pseudo_map * (1 - img_mask)        
                vae_mix_unlab_label = label_batch[:self.args.labeled_bs] * (1 - img_mask) + vae_pseudo_map * img_mask
                cnn_mix_lab_label = label_batch[:self.args.labeled_bs] * img_mask + cnn_pseudo_map * (1 - img_mask)        
                cnn_mix_unlab_label = label_batch[:self.args.labeled_bs] * (1 - img_mask) + cnn_pseudo_map * img_mask   
                
                vae_mix_label = torch.concat([vae_mix_lab_label, vae_mix_unlab_label], dim=0)
                cnn_mix_label = torch.concat([cnn_mix_lab_label, cnn_mix_unlab_label], dim=0)
                
                vae_mix_pred, cnn_mix_pred = self.model(img_rgb_mix, None, is_train=True)
                vae_pred_mixed_loss = self.seg_loss(vae_mix_pred, cnn_mix_label)
                cnn_pred_mixed_loss = self.seg_loss(cnn_mix_pred, vae_mix_label)
                mixed_loss = vae_pred_mixed_loss + cnn_pred_mixed_loss

                loss += consistency_weight * mixed_loss
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
        

            logger.info('iteration %d : '
                        '  loss : %f'
                        '  sup_loss : %f'
                        '  unsup_loss : %f'
                        '  lr_ : %6f'

                        % (iter_num, 
                            loss, 
                            sup_loss, 
                            consistency_weight * unsup_loss, 
                            current_lr
                            ))

    def val(self, val_loader, snapshot_path, logger, iter_num):
        
        if isinstance(self.model, torch.nn.DataParallel):
            self.model = self.model.module 
        self.model.eval()
        
        if self.args.stage == "1":
            test_loss = 0.0

            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(val_loader):
                    val_image, val_label = sampled_batch['image'].to(self.args.device), sampled_batch['label'].to(self.args.device)
                    if val_image.shape[1] == 1:
                        val_image = val_image.repeat(1, 3, 1, 1)
                    
                    img_rgb = 2. * val_image - 1.
                    seg_rgb = 2. * val_label - 1.
                    
                    vae_latent_mean, cnn_latent_mean, seg_latent_mean = self.model(img_rgb, seg_rgb, is_train=False)
                    # vae_latent_mean, seg_latent_mean = self.model(img_rgb, seg_rgb, is_train=False)
                    test_loss += F.mse_loss(vae_latent_mean, seg_latent_mean) + F.mse_loss(cnn_latent_mean, seg_latent_mean)
                    # test_loss += F.mse_loss(vae_latent_mean, seg_latent_mean)
                
            test_loss = test_loss / len(val_loader)
            
            logger.info('iteration %d : '
                        '  avg_test_loss : %f '
                        % (iter_num, test_loss))
            
            if test_loss < self.best_performance:
                self.best_performance = test_loss
                save_best = os.path.join(snapshot_path, 'best_model.pth')
                # save_best = os.path.join(snapshot_path, 'best_model_pretrain_' + str(iter_num) + '.pth')
                torch.save({
                    'resnet': self.model.resnet.state_dict(),
                    'mapping_model': self.model.mapping_model.state_dict(),
                }, save_best)
        
        else:
            avg_dice = 0.0
            avg_HD95 = 0.0

            for i_batch, sampled_batch in enumerate(val_loader):
                val_image, val_label = sampled_batch['image'].to(self.args.device), sampled_batch['label'].to(self.args.device)
                if val_image.shape[1] == 1:
                    val_image = val_image.repeat(1, 3, 1, 1)
                
                img_rgb = 2. * val_image - 1.
                vae_pred, cnn_pred = self.model(img_rgb, None, is_train=False)
                pred = (vae_pred + cnn_pred) / 2
                # pred_soft = torch.softmax(pred, dim=1)
                pred_soft = pred
                
                dice_output = dice_coef(val_label, pred_soft, thr=0.5)
                HD95_output = evaluate_95hd(val_label, pred_soft, thr=0.5)

                avg_dice += dice_output
                avg_HD95 += HD95_output


            avg_dice = avg_dice / len(val_loader)
            avg_HD95 = avg_HD95 / len(val_loader)


            logger.info('iteration %d : '
                        '  avg_dice : %f '
                        '  avg_HD95 : %f '

                        % (iter_num, avg_dice, avg_HD95))


            if avg_dice > self.best_performance:
                self.best_performance = avg_dice
                save_best = os.path.join(snapshot_path, 'best_model_' + str(iter_num) + '.pth')
                torch.save({
                    'model': self.model.state_dict(),
                }, save_best)
        
        self.model.train()
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0, 1])
        
