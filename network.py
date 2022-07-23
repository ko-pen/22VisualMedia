from email.mime.application import MIMEApplication
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5,
                 stride=1, padding="same", dilation=1, groups=1, bias=True):
        super().__init__()
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, False)
        self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
        nn.init.kaiming_normal_(self.Conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x, m = input
        x = x * m
        x = self.Conv(x)
        k = self.Conv.kernel_size[0]
        if x.get_device()==-1:
            weights = torch.ones(torch.Size([1, 1, k, k]))
        else:
            weights = torch.ones(torch.Size([1, 1, k, k])).to(x.get_device())
        m = F.conv2d(m, weights, bias=None, stride=self.Conv.stride, padding=self.Conv.padding, dilation=self.Conv.dilation)
        mc = torch.clamp(m, min=1e-5)
        mc = 1. / mc
        x = x * mc
        x = x + self.bias.view(1, self.bias.size(0), 1, 1).expand_as(x)
        m = (m > 0).float()
        #print("SparceConv", x.shape, m.shape, torch.min(x), torch.max(x))
        return x, m

class SparseMaxPool(nn.Module):
    def __init__(self, kernel_size=2, stride=None, padding=0, dilation=1):
        super().__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size, stride, padding, dilation)

    def forward(self, input):
        x, m = input
        x = x * m
        x = self.MaxPool(x)
        m = self.MaxPool(m)
        mc = torch.clamp(m, min=1e-5)
        mc = 1. / mc
        x = x * mc
        #print("SparceMaxPool", x.shape, m.shape, torch.min(x), torch.max(x))
        return x, m

class SparseUpsample(nn.Module):
    def __init__(self, scale_factor=2, mode='bilinear'):
        super().__init__()
        self.Upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def forward(self, input):
        x, m = input
        x = x * m
        x = self.Upsample(x)
        m = self.Upsample(m)
        mc = torch.clamp(m, min=1e-5)
        mc = 1. / mc
        x = x * mc
        m = (m > 0).float()
        #print("SparceUpsample", x.shape, m.shape, torch.min(x), torch.max(x))
        return x, m

class SparseSum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        x1, m1, x2, m2 = input
        x1 = x1 * m1
        x2 = x2 * m2
        x = x1 + x2
        m = m1 + m2
        mc = torch.clamp(m, min=1e-5)
        mc = 1. / mc
        x = x * mc
        m = (m > 0).float()
        #print("SparceSum", x.shape, m.shape, torch.min(x), torch.max(x))
        return x, m

class SparseConcatConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 stride=1, padding="same", dilation=1, groups=1, bias=True):
        super().__init__()
        self.Conv1 = nn.Conv2d(in_channels*2, out_channels, kernel_size, stride, padding, dilation, groups, False)
        self.Conv2 = nn.Conv2d(in_channels*2, out_channels, kernel_size, stride, padding, dilation, groups, False)
        self.Conv3 = nn.Conv2d(in_channels*2, out_channels, kernel_size, stride, padding, dilation, groups, False)
        self.bias = nn.Parameter(torch.zeros(out_channels).float(), requires_grad=True)
        nn.init.kaiming_normal_(self.Conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.Conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.Conv3.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x1, m1, x2, m2 = input
        x = torch.cat([x1,x2], dim=1)
        x1 = self.Conv1(x)
        x2 = self.Conv2(x)
        x3 = self.Conv3(x)
        x1 = (m1 > 0).float() * (m2 == 0).float() * x1
        x2 = (m1 == 0).float() * (m2 > 0).float() * x2
        x3 = (m1 > 0).float() * (m2 > 0).float()  * x3
        x = x1 + x2 + x3
        m = m1 + m2
        m = (m > 0).float()
        #print("SparceConcatConv", x.shape, m.shape, torch.min(x), torch.max(x))
        return x, m

class TwoScaleBlock(nn.Module):
    def __init__(self, k=5, middle_level_scale=2):
        super().__init__()
        self.SConv1 = SparseConv(16, 16, kernel_size=k)
        self.MaxPool = SparseMaxPool(kernel_size=middle_level_scale)
        self.CC = SparseConcatConv(16, 16)
        self.SConv2 = SparseConv(16, 16, kernel_size=k)
        self.Conv = nn.Conv2d(16, 16, kernel_size=1, padding='same')
        self.Upsample = SparseUpsample(scale_factor=middle_level_scale)
        self.Sum = SparseSum()
    
    def forward(self, input):
        #print("in Two")
        x_low, m_low, x_middle, m_middle = input

        x_low, m_low = self.SConv1((x_low, m_low))

        x_middle_, m_middle_ = self.MaxPool((x_low, m_low))
        x_middle, m_middle = self.CC((x_middle_, m_middle_, x_middle, m_middle))
        
        x_middle, m_middle = self.SConv2((x_middle, m_middle))

        x_middle_ = self.Conv(x_middle)
        
        x_middle_, m_middle_ = self.Upsample((x_middle_, m_middle))
        x_low, m_low = self.Sum((x_low, m_low, x_middle_, m_middle_))

        #print("SparceTwo", x_low.shape, m_low.shape, x_middle.shape, m_middle.shape)
        #print("out Two")
        return x_low, m_low, x_middle, m_middle

class ThreeScaleBlock(nn.Module):
    def __init__(self, k=5, middle_level_scale=2, high_level_scale=4):
        super().__init__()
        self.SConv1 = SparseConv(16, 16, kernel_size=k)
        self.MaxPool_M = SparseMaxPool(kernel_size=middle_level_scale)
        self.MaxPool_H = SparseMaxPool(kernel_size=high_level_scale)
        self.CC1 = SparseConcatConv(16, 16)
        self.CC2 = SparseConcatConv(16, 16)
        self.SConv2 = SparseConv(16, 16, kernel_size=k)
        self.SConv3 = SparseConv(16, 16, kernel_size=k)
        self.Conv1 = nn.Conv2d(16, 16, kernel_size=1, padding='same')
        self.Conv2 = nn.Conv2d(16, 16, kernel_size=1, padding='same')
        self.Upsample_M = SparseUpsample(scale_factor=middle_level_scale)
        self.Upsample_H = SparseUpsample(scale_factor=high_level_scale)
        self.Sum = SparseSum()
    
    def forward(self, input):
        #print("in Three")
        x_low, m_low, x_middle, m_middle, x_high, m_high = input
        
        x_low, m_low = self.SConv1((x_low, m_low))

        x_middle_, m_middle_ = self.MaxPool_M((x_low, m_low))
        x_middle, m_middle = self.CC1((x_middle_, m_middle_, x_middle, m_middle))
        
        x_middle_, m_middle_ = self.MaxPool_H((x_low, m_low))
        x_high, m_high = self.CC2((x_middle_, m_middle_, x_high, m_high))

        x_middle, m_middle = self.SConv2((x_middle, m_middle))

        x_middle_ = self.Conv1(x_middle)
        x_middle_, m_middle_ = self.Upsample_M((x_middle_, m_middle))
        x_low, m_low = self.Sum((x_low, m_low, x_middle_, m_middle_))

        x_high, m_high = self.SConv3((x_high, m_high))

        x_high_ = self.Conv2(x_high)
        x_high_, m_high_ = self.Upsample_H((x_high_, m_high))
        x_low, m_low = self.Sum((x_low, m_low, x_high_, m_high_))

        #print("SparceThree", x_low.shape, m_low.shape, x_middle.shape, m_middle.shape, x_high.shape, m_high.shape)
        #print("out Three")

        return x_low, m_low, x_middle, m_middle, x_high, m_high

class HMSNet(nn.Module):
    def __init__(self, RGB_input=False):
        super().__init__()
        if not RGB_input:
            self.initialSConv5x5 = SparseConv(1, 16, kernel_size=5)
            self.SConv5x5 = SparseConv(16, 16, kernel_size=5)
            self.SConv1x1_1 = SparseConv(16, 16, kernel_size=1)
            self.MaxPool = SparseMaxPool(kernel_size=2)
            self.TwoScaleBlock1 = TwoScaleBlock(k=5, middle_level_scale=2)
            self.SConv1x1_2 = SparseConv(16, 16, kernel_size=1)
            self.ThreeScaleBlock1 = ThreeScaleBlock(k=5, middle_level_scale=2, high_level_scale=4)
            self.ThreeScaleBlock2 = ThreeScaleBlock(k=5, middle_level_scale=4, high_level_scale=8)
            self.ThreeScaleBlock3 = ThreeScaleBlock(k=5, middle_level_scale=2, high_level_scale=4)
            self.Upsample = SparseUpsample(scale_factor=2)
            self.Sum = SparseSum()
            self.TwoScaleBlock2 = TwoScaleBlock(k=5, middle_level_scale=2)
            self.Conv3x3 = nn.Conv2d(16, 16, kernel_size=3, padding='same')
            self.Conv1x1 = nn.Conv2d(16, 1, kernel_size=1, padding='same')

    def forward(self, SparseDepth, Mask, img=None):
        #print("input", SparseDepth.shape, Mask.shape, torch.min(SparseDepth), torch.max(SparseDepth))

        x_low, m_low = self.initialSConv5x5([SparseDepth, Mask])
        x_low, m_low = self.SConv1x1_1((x_low, m_low))
        x_middle, m_middle = self.MaxPool((x_low, m_low))

        x_low, m_low, x_middle, m_middle = self.TwoScaleBlock1((x_low, m_low, x_middle, m_middle))
        
        x_low, m_low = self.SConv1x1_2((x_low, m_low))
        x_middle_, m_middle_ = self.MaxPool((x_low, m_low))
        x_high, m_high = self.MaxPool((x_middle, m_middle))

        x_low, m_low, x_middle, m_middle, x_high, m_high = self.ThreeScaleBlock1((x_low, m_low, x_middle_, m_middle_, x_high, m_high))
        
        x_middle, m_middle = self.MaxPool((x_middle, m_middle))
        x_high, m_high = self.MaxPool((x_high, m_high))

        x_low, m_low, x_middle, m_middle, x_high, m_high = self.ThreeScaleBlock2((x_low, m_low, x_middle, m_middle, x_high, m_high))
        
        x_middle, m_middle = self.Upsample((x_middle, m_middle))
        x_high, m_high = self.Upsample((x_high, m_high))

        x_low, m_low, x_middle, m_middle, x_high, m_high = self.ThreeScaleBlock3((x_low, m_low, x_middle, m_middle, x_high, m_high))
        
        x_middle, m_middle = self.Upsample((x_middle, m_middle))
        x_low, m_low = self.Sum((x_low, m_low, x_middle, m_middle))
        x_middle, m_middle = self.Upsample((x_high, m_high))

        x_low, m_low, x_middle, m_middle = self.TwoScaleBlock2((x_low, m_low, x_middle, m_middle))
        
        x_middle, m_middle = self.Upsample((x_middle, m_middle))
        x_low, m_low = self.Sum((x_low, m_low, x_middle, m_middle))
        x_low = self.Conv3x3(x_low)
        return self.Conv1x1(x_low)