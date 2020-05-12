import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self,channels,bias=False):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels,channels,kernel_size=3,padding=0,bias=bias),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels,channels,kernel_size=3,padding=0,bias=bias),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        out = x + self.conv_block(x)        
        return out

class AdaILN(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(AdaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1,channels,1,1))
        self.rho.data.fill_(0.9)
    
    def forward(self, x, gamma, beta):
        mean_I, var_I = torch.mean(x, dim=[2,3],keepdims=True),torch.var(x, dim=[2,3],keepdims=True)
        mean_L, var_L = torch.mean(x, dim=[1,2,3],keepdims=True),torch.var(x, dim=[1,2,3],keepdims=True)
        out_I = (x-mean_I)/torch.sqrt(var_I+self.eps)
        out_L = (x-mean_L)/torch.sqrt(var_L+self.eps)
        rho = self.rho.expand(x.shape[0],-1,-1,-1)
        out = rho*out_I + (1-rho)*out_L
        out = out*gamma.unsqueeze(2).unsqueeze(3)+beta.unsqueeze(2).unsqueeze(3)
        return out

class ILN(nn.Module):
    def __init__(self,channels,eps=1e-5):
        super(ILN,self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1,channels,1,1))
        self.gamma = Parameter(torch.Tensor(1,channels,1,1))
        self.beta = Parameter(torch.Tensor(1,channels,1,1))
        self.rho.data.fill_(0.9)
        self.gamma.data.fill_(0.9)
        self.beta.data.fill_(0.9)
    
    def forward(self, x):
        mean_I, var_I = torch.mean(x, dim=[2,3],keepdims=True),torch.var(x, dim=[2,3],keepdims=True)
        mean_L, var_L = torch.mean(x, dim=[1,2,3],keepdims=True),torch.var(x, dim=[1,2,3],keepdims=True)
        out_I = (x-mean_I)/torch.sqrt(var_I+self.eps)
        out_L = (x-mean_L)/torch.sqrt(var_L+self.eps)
        rho = self.rho.expand(x.shape[0],-1,-1,-1)
        gamma = self.gamma.expand(x.shape[0],-1,-1,-1)
        beta = self.beta.expand(x.shape[0],-1,-1,-1)
        out = rho*out_I + (1-rho)*out_L
        out = out*gamma + beta
        return out

class AdaResBlock(nn.Module):
    def __init__(self,channels,bias=False):
        super(AdaResBlock,self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=0,bias=bias)
        self.norm1 = AdaILN(channels)
        self.relu = nn.ReLU(True)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=0,bias=bias)
        self.norm2 = AdaILN(channels)

    def forward(self, x, gamma, beta):
        out = self.norm1(self.conv1(self.pad1(x)),gamma, beta)
        out = self.relu(out)
        out = self.norm2(self.conv2(self.pad2(out)),gamma, beta)
        return out+x

class ResnetGenerator(nn.Module):
    def __init__(self,in_channels,out_channels,ngf=64,n_blocks=6,img_size=256,light=False):
        super(ResnetGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        # Down Sampling
        DownBlock = [nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channels,ngf,kernel_size=7, stride=1,bias=False),
                    nn.InstanceNorm2d(ngf),
                    nn.ReLU(True)]
        for i in range(2):
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf*mult,ngf*mult*2,kernel_size=3,stride=2,bias=False),
                          nn.InstanceNorm2d(ngf*mult*2),
                          nn.ReLU(True)]
        mult = mult * 2
        for i in range(n_blocks):
            DownBlock += [ResBlock(ngf*mult)]
        
        # CAM
        self.gap_fc = nn.Linear(ngf*mult,1,bias=False)
        self.gmp_fc = nn.Linear(ngf*mult,1,bias=False)
        self.conv1x1 = nn.Conv2d(ngf*mult*2,ngf*mult,kernel_size=1,stride=1)
        self.relu = nn.ReLU(True)

        # Gamma, Beta Block
        if self.light:
            FC = [nn.Linear(ngf*mult,ngf*mult,bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf*mult,ngf*mult,bias=False),
                  nn.ReLU(True)]
        else:
            FC = [nn.Linear(img_size//mult * img_size//mult * ngf * mult, ngf*mult,bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf*mult,ngf*mult,bias=False),
                  nn.ReLU(True)]
        self.gamma = nn.Linear(ngf*mult,ngf*mult,bias=False)
        self.beta = nn.Linear(ngf*mult,ngf*mult,bias=False)

        # Upsampling 
        for i in range(n_blocks):
            setattr(self, "UpBlock_{}".format(i+1),AdaResBlock(ngf*mult,ngf*mult))
        UpBlock = []
        for i in range(2):
            mult = 2**(2-i)
            UpBlock += [nn.Upsample(scale_factor=2,mode='nearest'),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(ngf*mult,int(ngf*mult/2),kernel_size=3,bias=False),
                        nn.ReLU(True)]
        UpBlock += [nn.ReflectionPad2d(3),
                    nn.Conv2d(ngf,out_channels,kernel_size=7,bias=False),
                    nn.Tanh()]
        
        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.UpBlock = nn.Sequential(*UpBlock)

    def forward(self,x):
        x = self.DownBlock(x)
        
        gap = F.adaptive_avg_pool2d(x,1)
        gap_logit = self.gap_fc(gap.view(x.shape[0],-1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x,1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0],-1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit],dim=1)
        x = torch.cat([gap,gmp],dim=1)
        x = self.relu(self.conv1x1(x))

        heatmap = torch.sum(x,dim=1,keepdim=True)

        if self.light:
            x_ = F.adaptive_avg_pool2d(x,1)
            x_ = self.FC(x_.view(x_.shape[0],-1))
        else:
            x_ = self.FC(x.view(x.shape[0],-1))
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self,'UpBlock_{}'.format(i+1))(x,gamma,beta)
        out = self.UpBlock(x)
        return out, cam_logit, heatmap

class Discriminator(nn.Module):
    def __init__(self,in_channels,ndf=64,n_layers=5):
        super(Discriminator,self).__init__()
        DownBlock = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                     nn.Conv2d(in_channels, ndf,kernel_size=4,stride=2)
                 ),
                 nn.LeakyReLU(0.2,True)]
        for i in range(1,n_layers-2):
            mult = 2** (i-1)
            DownBlock += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                          nn.Conv2d(ndf*mult,ndf*mult*2,kernel_size=4,stride=2)
                      ),
                      nn.LeakyReLU(0.2,True)]
        mult = 2**(n_layers-2-1)
        DownBlock += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                      nn.Conv2d(ndf*mult,ndf*mult*2,kernel_size=4,stride=1)
                  ),
                  nn.LeakyReLU(0.2,True)]
        
        # CAM
        mult = 2**(n_layers-2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf*mult,1,bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf*mult,1,bias=False))
        self.conv1x1 = nn.Conv2d(ndf*mult*2,ndf*mult,kernel_size=1,stride=1)
        self.leaky_relu = nn.LeakyReLU(0.2,True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf*mult,1,kernel_size=4,stride=1,bias=False)
        )
        self.DownBlock = nn.Sequential(*DownBlock)
    
    def forward(self, x):
        x = self.DownBlock(x)

        gap = F.adaptive_avg_pool2d(x,1)
        gap_logit = self.gap_fc(gap.view(x.shape[0],-1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = F.adaptive_max_pool2d(x,1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0],-1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit],1)
        x = torch.cat([gap,gmp],1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = torch.sum(x,dim=1,keepdims=True)
        x = self.pad(x)
        out = self.conv(x)
        return out, cam_logit, heatmap

class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w