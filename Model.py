#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn


# In[4]:


class res_block(nn.Module) :
    def __init__(self , in_channel, out_channel=32) :
        super(res_block , self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = [3 ,3], padding='same')
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = [3 ,3], padding='same')
        self.leaky_Relu = nn.LeakyReLU(inplace = True)
        
    def forward(self , input_x) :
        x = self.conv1(self.leaky_Relu(self.conv1(input_x)))
        return x + input_x


# In[5]:


class generator(nn.Module) :
    def __init__(self , channel=32, num_blocks=4 ) :
        super(generator , self).__init__()
        self.conv1 = nn.Conv2d(3, channel, kernel_size = [7 ,7], padding='same')
        self.conv2 = nn.Conv2d(channel, channel*2, kernel_size = [3 ,3], padding='same' , stride=2)
        self.conv3 = nn.Conv2d(channel, channel*2, kernel_size = [3 ,3], padding='same' )
        self.conv4 = nn.Conv2d(channel, channel*4, kernel_size = [3 ,3], padding='same' , stride=2)
        self.conv5 = nn.Conv2d(channel, channel*4, kernel_size = [3 ,3], padding='same' )
        
        self.resblock = nn.Sequential(*[res_block(channel * 4, channel * 4) for i in range(num_blocks)])
        
        self.conv6 = nn.Conv2d(channel*4, channel*2, kernel_size = [3 ,3], padding='same' )
        self.conv7 = nn.Conv2d(channel*2, channel*2, kernel_size = [3 ,3], padding='same' )
        self.conv8 = nn.Conv2d(channel*2, channel, kernel_size = [3 ,3], padding='same' )
        self.conv9 = nn.Conv2d(channel, channel, kernel_size = [3 ,3], padding='same' )
        self.conv10 = nn.Conv2d(channel, 3, kernel_size = [7 ,7], padding='same' )
        
        self.leaky_Relu = nn.LeakyReLU(inplace = True)
        self.up_sampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.tanh_activation = nn.Tanh()
        
    def forward(self , input_x):
        x_1 = self.leaky_Relu(self.conv1(input_x))
        
        x_2 = self.leaky_Relu(self.conv2(x_1))
        x_2 = self.leaky_Relu(self.conv3(x_2))
        
        x_3 = self.leaky_Relu(self.conv4(x_2))
        x_3 = self.leaky_Relu(self.conv5(x_3))
        
        x_4 = self.resblock(x_3)
        x_4 = self.leaky_Relu(self.conv6(x_3))
        
        x_5 = self.up_sampling(x_4)
        x_5 = self.leaky_Relu(self.conv7(x_5 + x_2))
        x_5 = self.leaky_Relu(self.conv8(x_5))
        
        x_6 = self.up_sampling(x_5)
        x_6 = self.leaky_Relu(self.conv9(x_6 + x_1))
        x_6 = self.conv10(x_6)
        
        return self.tanh_activation(x_6)


# In[ ]:




