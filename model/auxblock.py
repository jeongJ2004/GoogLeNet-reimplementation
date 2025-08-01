import torch.nn as nn
from model.convblock import ConvBlock

class AuxBlock(nn.Module):

    def __init__(self, in_c, num_class):
        super().__init__()

        self.avgpool = nn.AvgPool2d(5, 3)
        self.conv1x1 = ConvBlock(in_c, 128, 1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Lienar(1024, num_class)

        self.dropout = nn.Dropout(0.7)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv1x1(x)

        x = x.reshape(x.shape[0], -1)
        x = x.relu(self.fc1(x))
        x = x.dropout(x)
        x = self.fc2(x)

        return x
    