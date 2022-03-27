#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch
import numpy as np
import math
from torch.nn import functional as F
from torch.nn.functional import pad
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from os.path import exists


# In[ ]:

class res_block(Module) :
    def __init__(self , in_channel, out_channel=32) :
        super(res_block , self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = [3 ,3], padding='same' )
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = [3 ,3], padding='same' )
        self.leaky_Relu = nn.LeakyReLU(negative_slope=0.2, inplace = True)
        
    def forward(self , input_x) :
        X03_res1 = self.conv1(input_x)
        X03_res1_1 = X03_res1.detach().numpy()
        X03_res1_1 = np.transpose(X03_res1_1,(0,2,3,1))
        #np.save('paper_X03_res_1_3.npy',X03_res1_1)
        x = self.conv2(self.leaky_Relu(X03_res1))
        x3_1 = x + input_x
        x3_1= x3_1.detach().numpy()
        x3_1 = np.transpose(x3_1,(0,2,3,1))
        #print(x0)
        #file_exists = exists('/content/paper_X03_res.npy')
        #if file_exists :
        #  file_exists = exists('/content/paper_X03_res1.npy')
        #  if file_exists :
        #    file_exists = exists('/content/paper_X03_res2.npy')
        #    if file_exists :
        #        np.save('paper_X03_res3.npy',x3_1)
        #    else :
        #      np.save('paper_X03_res2.npy',x3_1)
        #  else :
        #    np.save('paper_X03_res1.npy',x3_1)
        #else :
        #  np.save('paper_X03_res.npy',x3_1)
        return x + input_x

# In[ ]:

class _ConvNd(Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,padding, dilation, transposed, output_padding,
                 groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            #bound = math.sqrt( 6 / (fan_in + fan_out) )
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows - input_rows)
    padding_rows = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows)
    rows_odd = (padding_rows % 2 != 0)
    padding_cols = max(0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows)
    cols_odd = (padding_rows % 2 != 0)

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(input, weight, bias, stride,padding=(padding_rows // 2, padding_cols // 2),dilation=dilation, groups=groups)

class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,False, _pair(0), groups, bias)

    def forward(self, input):
        return conv2d_same_padding(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x

class generator(nn.Module) :
    def __init__(self , channel=32, num_blocks=4 ) :
        super(generator , self).__init__()
        self.conv1 = nn.Conv2d(3, channel, kernel_size = [7 ,7], padding=3 )
        self.conv2 = nn.Conv2d(channel, channel*2, kernel_size = [3 ,3], padding=1 , stride=2)
        self.conv3 = nn.Conv2d(channel, channel*2, kernel_size = [3 ,3], padding=1 )
        self.conv4 = nn.Conv2d(channel, channel*4, kernel_size = [3 ,3], padding=1 , stride=2)
        self.conv5 = nn.Conv2d(channel, channel*4, kernel_size = [3 ,3], padding=1 )
        
        self.resblock = nn.Sequential(*[res_block(channel * 4, channel * 4) for i in range(num_blocks)])
        
        self.conv6 = nn.ConvTranspose2d(channel*4, channel*2, kernel_size = [3 ,3], padding=1 ,stride=2)
        self.conv7 = nn.Conv2d(channel*2, channel*2, kernel_size = [3 ,3], padding=1 )
        self.conv8 = nn.ConvTranspose2d(channel*2, channel, kernel_size = [3 ,3], padding=1 ,stride=2)
        self.conv9 = nn.Conv2d(channel, channel, kernel_size = [3 ,3], padding=1 )
        self.conv10 = nn.Conv2d(channel, 3, kernel_size = [7 ,7], padding=3 )
        
        self.leaky_Relu = nn.LeakyReLU(inplace = True)
        
    def forward(self , input_x):
        x = self.leaky_Relu(self.conv1(input_x))
        
        x = self.conv2(x)
        x = self.leaky_Relu(self.conv3(x))
        
        x = self.conv4(x)
        x = self.leaky_Relu(self.conv5(x))
        
        x = self.resblock(x)
        
        x = self.conv6(x)
        x = self.leaky_Relu(self.conv7(x))
        
        x = self.conv8(x)
        x = self.leaky_Relu(self.conv9(x))
        
        x = self.conv10(x)
        
        return x


# In[ ]:

class unet_generator(nn.Module) :
    def __init__(self , channel=32, num_blocks=4 ) :
        super(unet_generator , self).__init__()
        self.conv1 = nn.Conv2d(3, channel, kernel_size = [7 ,7], padding='same' )
        self.conv2 = Conv2d(channel, channel, kernel_size = [3 ,3], padding='same' , stride=2)
        self.conv3 = nn.Conv2d(channel, channel*2, kernel_size = [3 ,3], padding='same' )
        self.conv4 = Conv2d(channel*2, channel*2, kernel_size = [3 ,3], padding='same' , stride=2)
        self.conv5 = nn.Conv2d(channel*2, channel*4, kernel_size = [3 ,3], padding='same' )
        
        self.resblock = nn.Sequential(*[res_block(channel * 4, channel * 4) for i in range(num_blocks)])
        #self.resblock1 = nn.Sequential(*[res_block(channel * 4, channel * 4) for i in range(1)])

        self.conv6 = nn.Conv2d(channel*4, channel*2, kernel_size = [3 ,3], padding='same' )
        self.conv7 = nn.Conv2d(channel*2, channel*2, kernel_size = [3 ,3], padding='same' )
        self.conv8 = nn.Conv2d(channel*2, channel, kernel_size = [3 ,3], padding='same' )
        self.conv9 = nn.Conv2d(channel, channel, kernel_size = [3 ,3], padding='same' )
        self.conv10 = nn.Conv2d(channel, 3, kernel_size = [7 ,7], padding='same' )
        
        self.leaky_Relu = nn.LeakyReLU(negative_slope=0.2, inplace = True)
        self.up_sampling = Interpolate(scale_factor=2, mode='bilinear')
        #self.tanh_activation = nn.Tanh()
        
    def forward(self , input_x):
        x_0 = self.conv1(input_x)
        #print(x_0)
        #x0= x_0.view(x_0.shape[0],x_0.shape[2],x_0.shape[3],x_0.shape[1])
        #x0= x_0.detach().numpy()
        #x0 = np.transpose(x0,(0,2,3,1))
        #print(x0)
        #np.save('paper_X01.npy',x0)
        x_1 = self.leaky_Relu(x_0)
        #x1= x_1.detach().numpy()
        #x1 = np.transpose(x1,(0,2,3,1))
        #print(x0)
        #np.save('paper_X01_relu.npy',x1)
        #print(x_1.shape)
        x_2 = self.leaky_Relu(self.conv2(x_1))
        #print(x_2.shape)
        x_2 = self.leaky_Relu(self.conv3(x_2))
        #print(x_2.shape)
        
        x_3 = self.leaky_Relu(self.conv4(x_2))
        #print(x_3.shape)
        x_3 = self.leaky_Relu(self.conv5(x_3))
        #print(x_3.shape)
        #x3= x_3.detach().numpy()
        #x3 = np.transpose(x3,(0,2,3,1))
        #print(x0)
        #np.save('paper_X03_relu.npy',x3)

        x_4 = self.resblock(x_3)
        #print(x_4.shape)
        #x4= x_4.detach().numpy()
        #x4 = np.transpose(x4,(0,2,3,1))
        #print(x0)
        #np.save('paper_X04.npy',x4)
        x_4 = self.leaky_Relu(self.conv6(x_4))
        #print(x_4.shape)
        
        #x5= x_4.detach().numpy()
        #x5 = np.transpose(x5,(0,2,3,1))
        #print(x0)
        #np.save('paper_X05.npy',x5)

        x_5 = self.up_sampling(x_4)
        #print(x_5.shape)

        #x6= x_5.detach().numpy()
        #x6 = np.transpose(x6,(0,2,3,1))
        #print(x0)
        #np.save('paper_X06.npy',x6)

        x_5 = self.leaky_Relu(self.conv7(x_5 + x_2))
        #print(x_5.shape)
        x_5 = self.leaky_Relu(self.conv8(x_5))
        #print(x_5.shape)
        
        #x7= x_5.detach().numpy()
        #x7 = np.transpose(x7,(0,2,3,1))
        #print(x0)
        #np.save('paper_X07.npy',x7)

        x_6 = self.up_sampling(x_5)
        #print(x_6.shape)
        x_6 = self.leaky_Relu(self.conv9(x_6 + x_1))
        #print(x_6.shape)

        #x8 = x_6.detach().numpy()
        #x8 = np.transpose(x8,(0,2,3,1))
        #print(x0)
        #np.save('paper_X08.npy',x8)

        x_6 = self.conv10(x_6)
        #print(x_6.shape)
        
        #self.tanh_activation(x_6)
        return x_6

# In[ ]:


class Discriminator_BatchNorm(nn.Module):
    def __init__(self, channel=32, is_training=True, patch=True):
        super(Discriminator_BatchNorm , self).__init__()
        self.channel = channel
        self.patch = patch
        self.track_running_stats = is_training
        in_channel = 3
        layers = []
        for idx in range(3) :
            layers.extend([
                Conv2d(in_channel,channel*(2**idx), kernel_size=[3,3], stride=2, padding='same'),
                nn.BatchNorm2d(track_running_stats=track_running_stats),
                nn.LeakyReLU(inplace = True),
                nn.Conv2d(channel*(2**idx),channel*(2**idx), kernel_size=[3,3], padding='same'),
                nn.BatchNorm2d(track_running_stats=track_running_stats),
                nn.LeakyReLU(inplace = True)
            ])
            
            in_channel = channel*(2**idx)
        
        self.body = nn.Sequential(*layers)
        if self.patch :
            self.head = nn.Conv2d(in_channel,1, kernel=[1,1], padding=0)
        else :
            self.head = nn.Sequential(torch.mean(dim=[1,2]), nn.Linear(in_channel,1))
            
    def forward(self , input_x):
        x = self.body(input_x)
        x = self.head(x)
        
        return x


# In[ ]:


class Discriminator_SpectralNorm(nn.Module):
    def __init__(self, scale =1 ,channel=32, patch=True ,input_channel=1):
        super(Discriminator_SpectralNorm , self).__init__()
        self.channel = channel
        self.patch = patch
        self.scale = scale
        self.input_channel = input_channel
        layers = []
        for idx in range(3) :
            layers.extend([
                spectral_norm(Conv2d(self.input_channel,channel*(2**idx), kernel_size=[3,3], stride=2, padding='same')),
                nn.LeakyReLU(negative_slope=0.2, inplace = True),
                spectral_norm(nn.Conv2d(channel*(2**idx),channel*(2**idx), kernel_size=[3,3], padding='same')),
                nn.LeakyReLU(negative_slope=0.2, inplace = True)
            ])
            
            self.input_channel = channel*(2**idx)
        
        self.body = nn.Sequential(*layers)
        #print(self.body)
        #print(self.patch)
        if self.patch :
            self.head = spectral_norm(nn.Conv2d(self.input_channel,1, kernel_size=[1,1], padding=0))
        else :
            self.head = nn.Sequential(torch.mean(dim=[1,2]), nn.Linear(self.input_channel,1))
        #print(self.head)
            
    def forward(self , input_x):
        x = self.body(input_x)
        x = self.head(x)
        
        return x


# In[ ]:


class Discriminator_LayerNorm(nn.Module):
    def __init__(self, channel=32, patch=True):
        super(Discriminator_LayerNorm , self).__init__()
        self.channel = channel
        self.patch = patch
        in_channel = 3
        layers = []
        for idx in range(3) :
            layers.extend([
                nn.Con2d(in_channel,channel*(2**idx), kernel=[3,3], stride=2, padding=1),
                nn.LayerNorm(),
                nn.LeakyReLU(inplace = True),
                nn.Con2d(channel*(2**idx),channel*(2**idx), kernel=[3,3], padding=1),
                nn.LayerNorm(),
                nn.LeakyReLU(inplace = True)
            ])
            
            in_channel = channel*(2**idx)
        
        self.body = nn.Sequential(*layers)
        if self.patch :
            self.head = nn.Con2d(in_channel,1, kernel=[1,1], padding=0)
        else :
            self.head = nn.Sequential(torch.mean(dim=[1,2]), nn.Linear(in_channel,1))
            
    def forward(self , input_x):
        x = self.body(input_x)
        x = self.head(x)
        
        return x


# In[ ]:




