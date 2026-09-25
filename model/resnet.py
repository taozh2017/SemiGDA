import torch
from torch import Tensor
import torch.nn as nn
from collections import OrderedDict
import re
import numpy as np
import torchvision.models as models

import torch.nn.functional as F
import torch.utils.checkpoint as cp
from typing import Type, Any, Callable, Union, List, Optional, cast, Tuple
from torch.distributions.uniform import Uniform



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
           
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        #x = x1 + x2
        x = self.conv_bn_relu(x)
        return x  


class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
    
class ConvBlock(nn.Module):
    """two convolution layers with batch norm and leaky relu"""

    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv_conv(x)

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, zero_init_residual=False,groups = 1, width_per_group = 64): 
        super(ResNet, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 128
        self.dilation = 1
        replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1)
        # self.conv1 = nn.Conv2d(self.inplanes, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer2 = self._make_layer(block, 512, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilate=replace_stride_with_dilation[2])
        self.conv_out = ConvBlock(512, 8, 0.1)
        
        # channels = [64,64,128,256,512]
        # self.decode4 = Decoder(channels[4],channels[3])
        # self.decode3 = Decoder(channels[3],channels[2])
        # self.decode2 = Decoder(channels[2],channels[1])
        # self.decode1 = Decoder(channels[1],channels[0])
        # self.decode0 = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
        #     nn.Conv2d(32, num_classes, kernel_size=1,bias=False))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0) 

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        encoder = []
        
        x = self.conv1(x)
                
        x = self.bn1(x)
        x = self.relu(x)
        encoder.append(x)
        
        # x = self.maxpool(x)
        x = self.layer1(x)
        encoder.append(x)
        
        x = self.layer2(x)
        encoder.append(x)
        
        x = self.layer3(x)
        encoder.append(x)
        
        # x = self.layer4(x)
        # encoder.append(x)
        
        x = self.conv_out(x)
        # d4 = self.decode4(encoder[4], encoder[3]) 
        # d3 = self.decode3(d4, encoder[2]) 
        # d2 = self.decode2(d3, encoder[1]) 
        # d1 = self.decode1(d2, encoder[0]) 
        # out = self.decode0(d1)    
        # return out
        return x, encoder

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def resnet34(num_class) :
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes = num_class)