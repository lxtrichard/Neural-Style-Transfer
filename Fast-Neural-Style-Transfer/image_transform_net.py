import torch
import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride):
        super(ConvLayer,self).__init__()
        padding = kernel_size // 2
        self.reflect_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x):
        x = self.reflect_pad(x)
        out = self.conv2d(x)
        return out

class UpsampleConvLayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,up_sample=None):
        super(UpsampleConvLayer, self).__init__()
        self.up_sample = up_sample
        padding = kernel_size // 2
        self.reflect_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels,out_channels,kernel_size,stride)
    
    def forward(self, x):
        if self.up_sample:
            x = nn.functional.interpolate(x, mode='nearest',scale_factor=self.up_sample)
        x = self.reflect_pad(x)
        out = self.conv2d(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels,channels,kernel_size=3,stride=1)
        self.IN1 = nn.InstanceNorm2d(channels,affine=True)
        self.conv2 = ConvLayer(channels,channels,kernel_size=3,stride=1)
        self.IN2 = nn.InstanceNorm2d(channels,affine=True)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        res = x
        x = self.relu(self.IN1(self.conv1(x)))
        x = self.relu(self.IN2(self.conv2(x)))
        out = x + res
        return out

class ImageTransformNet(nn.Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()
        self.relu = nn.ReLU()
        # Encoding Layer
        self.conv1 = ConvLayer(3,32,kernel_size=3,stride=1)
        self.IN1 = nn.InstanceNorm2d(32,affine=True)
        self.conv2 = ConvLayer(32,64,kernel_size=3,stride=2)
        self.IN2 = nn.InstanceNorm2d(64,affine=True)
        self.conv3 = ConvLayer(64,128,kernel_size=3,stride=2)
        self.IN3 = nn.InstanceNorm2d(128,affine=True)
        # Residual Layer
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Decoding Layer
        self.deconv3 = UpsampleConvLayer(128,64,kernel_size=3, stride=1, up_sample=2)
        self.IN4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64,32,kernel_size=3, stride=1, up_sample=2)
        self.IN5 = nn.InstanceNorm2d(32,affine=True)
        self.deconv1 = UpsampleConvLayer(32,3,kernel_size=9, stride=1)
    
    def forward(self, x):
        x = self.relu(self.IN1(self.conv1(x)))
        x = self.relu(self.IN2(self.conv2(x)))
        x = self.relu(self.IN3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.relu(self.IN4(self.deconv3(x)))
        x = self.relu(self.IN5(self.deconv2(x)))
        x = self.deconv1(x)
        return x