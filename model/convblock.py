import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, **kwargs): #kwargs is for like stride = 2, padding = 1 ... it automatically creates more params
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
    