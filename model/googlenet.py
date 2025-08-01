import torch.nn as nn 
from model.convblock import ConvBlock
from model.inceptionblock import InceptionBlock
from model.auxblock import AuxBlock

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.conv1 = ConvBlock(3, 64, 7, 2, padding=3)
        self.conv2 = ConvBlock(64, 192, 3, 2, padding=1)
        
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.avgpool = nn.AvgPool2d(7, 1)

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.aux1 = AuxBlock(512, num_classes)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.aux2 = AuxBlock(528, num_classes)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        

        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)


    def forward(self, x):
        y = None
        z = None 

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)

        x = self.inception4a(x)
        y = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        z = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)

        x = self.fc(x)

        return x, y, z

def main():
    import torch

    model = GoogLeNet(10)
    print(model)

if __name__ == "__main__":
    main()
