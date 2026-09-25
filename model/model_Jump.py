import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from model.resnet import resnet34
from model.VAE_Jump import AutoencoderKL
from model.latent_mapping_model import ResAttnUNet_DS
# from model.UNet import Encoder
from model.distributions import DiagonalGaussianDistribution
import copy


def get_vae_encoding_mu_and_sigma(encoder_posterior, scale_factor):
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        mean, logvar = encoder_posterior.mu_and_sigma()
    else:
        raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
    return scale_factor * mean, logvar



class Semi_GMS(nn.Module):
    def __init__(self, args, in_chns, class_num, stage="", bilinear=True):
        super(Semi_GMS, self).__init__()
        
        self.args = args
        self.num_classes = class_num
        self.stage = stage
        self.ds_list = ['level2', 'level1', 'out']
        
        # get VAE (first-stage model)
        vae_path = './model/SD-VAE-weights/v2-inference-v-first-stage-VAE.yaml'
        vae_config = OmegaConf.load(f"{vae_path}")
        self.vae_model = AutoencoderKL(**vae_config.first_stage_config.get("params", dict()))
        
        self.scale_factor = vae_config.first_stage_config.scale_factor
        
        self.resnet = resnet34(class_num)
        self.mapping_model = ResAttnUNet_DS(in_channel=4, out_channels=4, num_res_blocks=2, ch=32, ch_mult=[1,2,4,4])
        
        self.seg_head = nn.Sequential(
                            nn.Conv2d(3, 16, kernel_size=3, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(16, self.args.num_classes, kernel_size=3, padding=1),
                            nn.ReLU(),
                        )
        
        pl_sd = torch.load("./model/SD-VAE-weights/768-v-ema-first-stage-VAE.ckpt", map_location="cpu", weights_only=True)
        sd = pl_sd["state_dict"]
        self.vae_model.load_state_dict(sd, strict=False)
        
        if stage == "1":
            for name, param in self.vae_model.named_parameters():
                param.requires_grad = False
        
        elif stage == "2":
            # Step 3: 使用 strict=False 跳过未找到的层
            missing_keys, unexpected_keys = self.vae_model.load_state_dict(sd, strict=False)

            # 打印缺失和多余的层名，便于调试
            print("Missing keys:", missing_keys)
            print("Unexpected keys:", unexpected_keys)

            # Step 4: 将缺失的层设置为可学习
            for name, param in self.vae_model.named_parameters():
                if name in missing_keys:
                    print(f"Setting {name} to be trainable")
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            ckpt = torch.load(self.args.snapshot_path + "/best_model.pth", map_location="cpu", weights_only=True)
            map_ckpt = ckpt['mapping_model']
            res_ckpt = ckpt['resnet']
            
            self.mapping_model.load_state_dict(map_ckpt, strict=True)
            # for name, param in self.mapping_model.named_parameters():
            #     param.requires_grad = False
            self.resnet.load_state_dict(res_ckpt, strict=True)
            # for name, param in self.resnet.named_parameters():
            #     param.requires_grad = False
                

    def vae_decode(self, pred_mean, hs, is_img=False):
        z = 1. / self.scale_factor * pred_mean
        pred = self.vae_model.decode(z, hs)
        pred = torch.mean(pred, dim=1, keepdim=True)
        pred = torch.clamp((pred + 1.0) / 2.0, min=0.0, max=1.0) * (self.args.num_classes - 1)
        # if self.args.stage == "1":
        #     if is_img:
        #         pred = torch.clamp((pred + 1.0) / 2.0, min=0.0, max=1.0)
        #     else:
        #         pred = torch.mean(pred, dim=1, keepdim=True)
        #         pred = torch.clamp((pred + 1.0) / 2.0, min=0.0, max=1.0) # (B, 1, H, W)
        # else:
        # pred = self.seg_head(pred)

        return pred
    
    def forward_stage1(self, image, mask):
        if image.shape[1] == 1:
            img_rgb = image.repeat(1, 3, 1, 1)
        else:
            img_rgb = image
            
        # if mask.shape[1] == 1:
        seg_rgb = mask.unsqueeze(1).repeat(1, 3, 1, 1)
        
        
        vae_posterior_img, vae_hs_img = self.vae_model.encode(img_rgb)
        posterior_seg, hs_seg = self.vae_model.encode(seg_rgb)
        img_latent_mean, img_latent_logvar = get_vae_encoding_mu_and_sigma(vae_posterior_img, self.scale_factor)
        seg_latent_mean, seg_latent_logvar = get_vae_encoding_mu_and_sigma(posterior_seg, self.scale_factor)
        
        vae_img_out_latent_mean = img_latent_mean
        vae_img_out_latent_mean_dict = self.mapping_model(vae_img_out_latent_mean)
        vae_latent_mean = vae_img_out_latent_mean_dict['out']
                
        cnn_h, feats_list = self.resnet(img_rgb)
        cnn_moments = self.vae_model.quant_conv(cnn_h)
        cnn_posterior = DiagonalGaussianDistribution(cnn_moments)
        cnn_latent_mean, cnn_latent_logvar = get_vae_encoding_mu_and_sigma(cnn_posterior, self.scale_factor)
        
        # vae_pred = self.vae_decode(vae_latent_mean, copy.deepcopy(vae_hs_img), is_img=False)
        # cnn_pred = self.vae_decode(cnn_latent_mean, copy.deepcopy(vae_hs_img), is_img=False)
        
        return vae_latent_mean, cnn_latent_mean, seg_latent_mean
    
    def forward_stage2(self, image, is_train=True):

        if image.shape[1] == 1:
            img_rgb = image.repeat(1, 3, 1, 1)
        else:
            img_rgb = image
    
        # with torch.no_grad():
        vae_posterior_img, vae_hs_img = self.vae_model.encode(img_rgb)
        img_latent_mean, img_latent_logvar = get_vae_encoding_mu_and_sigma(vae_posterior_img, self.scale_factor)

        cnn_h, feats_list = self.resnet(img_rgb)
        cnn_moments = self.vae_model.quant_conv(cnn_h)
        cnn_posterior = DiagonalGaussianDistribution(cnn_moments)
        cnn_latent_mean, cnn_latent_logvar = get_vae_encoding_mu_and_sigma(cnn_posterior, self.scale_factor)
        
        # get latent
        if is_train:
            img_latent_std = torch.exp(0.5 * img_latent_logvar)
            if np.random.uniform() > 0.5:
                img_latent_mean_aug = img_latent_mean + img_latent_std * torch.randn_like(img_latent_std)
            else:
                img_latent_mean_aug = img_latent_mean
                
            vae_img_out_latent_mean = img_latent_mean_aug
        else:
            vae_img_out_latent_mean = img_latent_mean
            
        vae_img_out_latent_mean_dict = self.mapping_model(vae_img_out_latent_mean)
        
        # vae_img_seg_dict = {}
        # for level_name in self.ds_list:
        # vae_pred = self.vae_decode(vae_img_out_latent_mean_dict['out'], copy.deepcopy(vae_hs_img), is_img=False)
        # cnn_pred = self.vae_decode(cnn_latent_mean, [feat.clone() for feat in feats_list], is_img=False)
        vae_pred = self.vae_decode(vae_img_out_latent_mean_dict['out'], vae_hs_img, is_img=False)
        cnn_pred = self.vae_decode(cnn_latent_mean, feats_list, is_img=False)
        return vae_pred, cnn_pred
    
    def forward(self, image, mask, is_train=True):
        
        if self.stage == "1":
            vae_latent_mean, cnn_latent_mean, seg_latent_mean = self.forward_stage1(image, mask)
            return vae_latent_mean, cnn_latent_mean, seg_latent_mean
        else:
            vae_pred, cnn_pred = self.forward_stage2(image, is_train)
            return vae_pred, cnn_pred



                