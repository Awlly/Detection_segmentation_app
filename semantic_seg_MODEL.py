import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class SemUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SemUNet, self).__init__()
        
        def CBR(in_feat, out_feat, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True)
            )
        
        # Contracting path
        self.enc1 = nn.Sequential(CBR(in_channels, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = nn.Sequential(CBR(128, 256), CBR(256, 256))
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = nn.Sequential(CBR(256, 512), CBR(512, 512))
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(CBR(512, 1024), CBR(1024, 1024))
        
        # Expansive path
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(CBR(1024, 512), CBR(512, 512))
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(CBR(512, 256), CBR(256, 256))
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(CBR(256, 128), CBR(128, 128))
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(CBR(128, 64), CBR(64, 64))
        
        # Output
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Contracting path
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Expansive path
        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, e4], dim=1))
        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e3], dim=1))
        u3 = self.up3(d2)
        d3 = self.dec3(torch.cat([u3, e2], dim=1))
        u4 = self.up4(d3)
        d4 = self.dec4(torch.cat([u4, e1], dim=1))
        
        # Output
        out = self.out_conv(d4)
        return out