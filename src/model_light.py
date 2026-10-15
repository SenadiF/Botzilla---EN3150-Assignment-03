import torch.nn as nn
from .DepthwiseSeparableConv import DepthwiseSeparableConv

# CNN - Depth wise separable : 100k parameters
class WasteImageNNLight(nn.Module):
    def __init__(self, num_classes=9, hidden=206):
        super().__init__()

        # features 
        self.features = nn.Sequential(
            # 3 -> 32
            DepthwiseSeparableConv(3,32),
            nn.MaxPool2d(2), # --> 32*32*32

            # 32 -> 64
            DepthwiseSeparableConv(32,64),
            nn.MaxPool2d(2), # --> 64*16*16

            # 64 -> 128
            DepthwiseSeparableConv(64, 128),
            nn.MaxPool2d(2), # --> 128*8*8

            # # 128 -> 256
            # DepthwiseSeparableConv(128, 256),
            # nn.MaxPool2d(2) # --> 256*4*4
        )

        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # adaptive avg pooling before flattern
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 256), # parameter reduction | parameters = input*hidden + hidden
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))