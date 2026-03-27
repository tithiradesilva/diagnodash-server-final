# model.py
# Architecture: MobileNetV3 (Backbone) + TCB/CBAM (Attention) + RefineDet (Dual-Stage Detection)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# --- ATTENTION MECHANISMS ---
# Channel Attention: Tells the network "WHAT" is important by pooling feature maps 
# and compressing them through a multi-layer perceptron (MLP).
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        # Squeeze spatial dimensions to 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Bottleneck architecture to learn cross-channel interactions
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

# Spatial Attention: Tells the network "WHERE" the object is by highlighting 
# critical regions in the feature map using a 7x7 convolution.
class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False) 
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        # Pool across the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True) 
        max_out, _ = torch.max(x, dim=1, keepdim=True) 
        
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x) 
        return self.sigmoid(x) 

# CBAM: Combines Channel and Spatial attention sequentially.
class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels) 
        self.sa = SpatialAttention() 

    def forward(self, x):
        x = x * self.ca(x) # Apply Channel Attention
        x = x * self.sa(x) # Apply Spatial Attention
        return x 

# Transfer Connection Block (TCB): Bridges the backbone features to the detection layers.
# It refines the features and applies the CBAM attention to focus on small dashboard icons.
class TransferConnectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_channels) 

    def forward(self, x):
        x = self.conv(x) 
        x = self.cbam(x) 
        return x

# --- MAIN ARCHITECTURE ---
class MobileNetRefineDetLiteCBAM(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes 
        self.num_anchors = 4 # Default anchors per spatial location

        # Lightweight Backbone
        backbone = models.mobilenet_v3_large(weights="DEFAULT") 
        self.features = backbone.features  

        # Multi-scale feature refinement blocks
        self.tcb1 = TransferConnectionBlock(40, 64) 
        self.tcb2 = TransferConnectionBlock(112, 64) 
        self.tcb3 = TransferConnectionBlock(160, 64) 

        # Smoothing layer used during Top-Down Feature Pyramid fusion
        self.smooth = nn.Conv2d(64, 64, 3, padding=1)

        # ARM (Anchor Refinement Module): Stage 1 - Filters background and adjusts initial anchors
        self.arm_loc_layers = nn.ModuleList([
            nn.Conv2d(64, self.num_anchors * 4, 3, padding=1),
            nn.Conv2d(64, self.num_anchors * 4, 3, padding=1),
            nn.Conv2d(64, self.num_anchors * 4, 3, padding=1)
        ])
        self.arm_conf_layers = nn.ModuleList([
            nn.Conv2d(64, self.num_anchors * 2, 3, padding=1), # 2 classes: Object vs Background
            nn.Conv2d(64, self.num_anchors * 2, 3, padding=1),
            nn.Conv2d(64, self.num_anchors * 2, 3, padding=1)
        ])

        # ODM (Object Detection Module): Stage 2 - Final classification and precise box regression
        self.odm_loc_layers = nn.ModuleList([
            nn.Conv2d(64, self.num_anchors * 4, 3, padding=1),
            nn.Conv2d(64, self.num_anchors * 4, 3, padding=1),
            nn.Conv2d(64, self.num_anchors * 4, 3, padding=1)
        ])
        self.odm_conf_layers = nn.ModuleList([
            nn.Conv2d(64, self.num_anchors * num_classes, 3, padding=1),
            nn.Conv2d(64, self.num_anchors * num_classes, 3, padding=1),
            nn.Conv2d(64, self.num_anchors * num_classes, 3, padding=1)
        ])

    def forward(self, x):
        sources = []
        
        # Extract features at different strides (8, 16, 32)
        for i, layer in enumerate(self.features):
            x = layer(x) 
            if i == 6:   
                sources.append(x)
            if i == 12:  
                sources.append(x)
            if i == 15:  
                sources.append(x)

        # Refine features with Attention (TCB + CBAM)
        f1 = self.tcb1(sources[0]) 
        f2 = self.tcb2(sources[1]) 
        f3 = self.tcb3(sources[2]) 

        # Feature Pyramid Network (FPN) - Top-Down Fusion
        # Injects strong semantic context from deep layers into shallow, high-resolution layers
        f3_up = F.interpolate(f3, size=f2.shape[2:], mode="bilinear", align_corners=False)
        f2 = self.smooth(f2 + f3_up) 

        f2_up = F.interpolate(f2, size=f1.shape[2:], mode="bilinear", align_corners=False)
        f1 = self.smooth(f1 + f2_up) 

        features = [f1, f2, f3] 

        # Stage 1: Anchor Refinement Predictions
        arm_loc = []
        arm_conf = []
        for f, loc_layer, conf_layer in zip(features, self.arm_loc_layers, self.arm_conf_layers):
            arm_loc.append(loc_layer(f).permute(0,2,3,1).contiguous())
            arm_conf.append(conf_layer(f).permute(0,2,3,1).contiguous())

        # Flatten tensors for loss calculation
        arm_loc = torch.cat([o.view(o.size(0), -1, 4) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1, 2) for o in arm_conf], 1)

        # Stage 2: Final Object Detection Predictions
        odm_loc = []
        odm_conf = []
        for f, loc_layer, conf_layer in zip(features, self.odm_loc_layers, self.odm_conf_layers):
            odm_loc.append(loc_layer(f).permute(0,2,3,1).contiguous())
            odm_conf.append(conf_layer(f).permute(0,2,3,1).contiguous())

        odm_loc = torch.cat([o.view(o.size(0), -1, 4) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in odm_conf], 1)

        return arm_loc, arm_conf, odm_loc, odm_conf