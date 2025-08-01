import torch.nn as nn
from model.convblock import ConvBlock
import torch

class InceptionBlock(nn.Module):
    
    def __init__(self, im_c, num_1x1, num_3x3_reduced, num_3x3, num_5x5_reduced, num_5x5, num_pool_proj):
        super().__init__()

        self.one_by_one = ConvBlock(im_c, num_1x1, kernel_size=1)

        self.three_by_three_redu = ConvBlock(im_c, num_3x3_reduced, kernel_size=1)
        self.three_by_three = ConvBlock(im_c, num_3x3, kernel_size=3, padding=1)

        self.five_by_five_redu = ConvBlock(im_c, num_5x5_reduced, kernel_size=1)
        self.five_by_five = ConvBlock(im_c, num_5x5, kernel_size=5, padding=1)

        self.maxpool = nn.MaxPool2d(3, 1, 1)
        self.pool_proj = ConvBlock(im_c, num_pool_proj, kernel_size=1)

    
    def forward(self, x):
        x1 = self.one_by_one(x)

        x2 = self.three_by_three_redu(x)
        x2 = self.three_by_three(x2)

        x3 = self.five_by_five_redu(x)
        x3 = self.five_by_five(x3)

        x4 = self.maxpool(x)
        x4 = self.pool_proj(x4)

        x = torch.cat([x1, x2, x3, x4], 1) #DepthConcat 
        return x
    


