import torch.nn as nn 
from model.convblock import ConvBlock
from model.inceptionblock import InceptionBlock
from model.auxblock import AuxBlock

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        self.stem = nn.Sequential(
            ConvBlock(3, 64, 7, 2, padding=3),
            nn.MaxPool2d(3, 2, 1),
            ConvBlock(64, 192, 3, 2, padding=1),
            nn.MaxPool2d(3, 2, 1)
        )

        self.inception3_cfg = [
            (192, 64, 96, 128, 16, 32, 32),
            (256, 128, 128, 192, 32, 96, 64)
        ]

        self.inception4_cfg = [
            (480, 192, 96, 208, 16, 48, 64),
            (512, 160, 112, 224, 24, 64, 64),
            (512, 128, 128, 256, 24, 64, 64),
            (512, 112, 144, 288, 32, 64, 64),
            (528, 256, 160, 320, 32, 128, 128)
        ]   

        self.inception5_cfg = [
            (832, 256, 160, 320, 32, 128, 128),
            (832, 384, 192, 384, 48, 128, 128)
        ]

        # Instantiate Inception blocks via loops
        self.inception3 = nn.ModuleList([InceptionBlock(*c) for c in self.inception3_cfg])
        self.inception4 = nn.ModuleList([InceptionBlock(*c) for c in self.inception4_cfg])
        self.inception5 = nn.ModuleList([InceptionBlock(*c) for c in self.inception5_cfg])

        

        # Aux classifiers 
        if aux_logits:
            self.aux1 = AuxBlock(512, num_classes) # After inception4_cfg[0] -> 4a
            self.aux2 = AuxBlock(528, num_classes) # After inception4_cfg[3] -> 4d

        # Pooling and Head
        self.maxpool = nn.Maxpool2d(3, 2, 1)
        self.avgpool = nn.AvgPool2d(7, 1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)



    def forward(self, x):
        aux1 = None
        aux2 = None 


        x = self.stem(x)

        for block in self.inception3:
            x = block(x)
        x = self.maxpool(x)

        for i, block in enumerate(self.inception4):
            x = block(x)
            if self.training and self.aux_logits:
                if i == 0:
                    aux1 = self.aux1(x)
                elif i == 3:
                    aux2 = self.aux2(x)
            x = self.maxpool(x)

        for block in self.inception5:
            x = block(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)

        x = self.fc(x)

        return (x, aux1, aux2) if (self.training and self.aux_logits) else x

def main():
    import torch

    model = GoogLeNet(10)
    print(model)

if __name__ == "__main__":
    main()
