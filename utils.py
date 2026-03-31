# utils.py
# Contains bounding box mathematics, Anchor generation, and IoU calculations.

import torch
import math
from itertools import product

# Generates the grid of default bounding boxes (anchors) across the image
class AnchorGenerator:
    def __init__(self, img_size):
        self.img_size = img_size
        
        # Matches the strides from the backbone (8, 16, 32)
        self.feature_maps = [
            img_size//8,
            img_size//16,
            img_size//32
        ]
        self.steps=[8,16,32] 
        self.min_sizes=[32,64,128] # Scaled specifically for small dashboard icons
        self.aspect_ratios=[1.0,1.5] # Square and slightly rectangular boxes

    def forward(self, device):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                # Calculate normalized center points [0, 1]
                cx = (j + 0.5) / f
                cy = (i + 0.5) / f
                
                s = self.min_sizes[k] / self.img_size 
                anchors.append([cx, cy, s, s])

                # Generate a slightly larger secondary square anchor
                s_prime = math.sqrt(s * (self.min_sizes[k] * 1.2 / self.img_size))
                anchors.append([cx, cy, s_prime, s_prime])

                # Generate rectangular anchors based on aspect ratios
                for ar in self.aspect_ratios:
                    if ar == 1:
                        continue 
                    anchors.append([cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)])
                    anchors.append([cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)])

        anchors = torch.tensor(anchors) 
        anchors.clamp_(0, 1) # Ensure boxes do not go out of image bounds
        return anchors.to(device)

# Converts Center-X, Center-Y, Width, Height -> X-min, Y-min, X-max, Y-max
def cxcywh_to_xyxy(boxes):
    x1y1 = boxes[:, :2] - boxes[:, 2:] / 2
    x2y2 = boxes[:, :2] + boxes[:, 2:] / 2
    return torch.cat([x1y1, x2y2], dim=1)

# Converts X-min, Y-min, X-max, Y-max -> Center-X, Center-Y, Width, Height
def xyxy_to_cxcywh(boxes):
    cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2
    wh   =  boxes[:, 2:] - boxes[:, :2]
    return torch.cat([cxcy, wh], dim=1)

# Calculates the overlapping area between two bounding boxes
def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    
    # Find the coordinates of the overlapping rectangle
    max_xy = torch.min(
        box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
        box_b[:, 2:].unsqueeze(0).expand(A, B, 2)
    )
    min_xy = torch.max(
        box_a[:, :2].unsqueeze(1).expand(A, B, 2),
        box_b[:, :2].unsqueeze(0).expand(A, B, 2)
    )
    
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

# Calculates Intersection over Union (IoU)
def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1)
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0)
              
    union = area_a + area_b - inter
    return inter / union

# Encodes Ground Truth boxes into mathematical offsets relative to the default anchors
def encode(matched, priors):
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    g_cxcy /= priors[:, 2:]
    
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh)
    return torch.cat([g_cxcy, g_wh], 1)

# Decodes the neural network's predicted offsets back into standard bounding box coordinates
def decode(loc, priors):
    cxcy = priors[:, :2] + loc[:, :2] * priors[:, 2:] 
    wh   = priors[:, 2:] * torch.exp(loc[:, 2:]) 
    
    x1y1 = cxcy - wh / 2
    x2y2 = cxcy + wh / 2
    return torch.cat([x1y1, x2y2], dim=1)