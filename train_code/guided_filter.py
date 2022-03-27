#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import numpy as np


# In[2]:


def tf_box_filter(x, r):
    #print(x.shape)
    k_size = int(2*r+1)
    ch = list(x.size())[1]
    weight = 1/(k_size**2)
    box_kernel = weight*np.ones((ch, 1, k_size, k_size))
    box_kernel = np.array(box_kernel).astype(np.float32)
    #print(k_size,ch,weight,box_kernel.shape)
    #box_kernel_temp = box_kernel.detach().numpy()
    #box_kernel_temp = np.transpose(box_kernel,(2,3,0,1))
    #output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
    conv = nn.Conv2d(x.shape[1],box_kernel.shape[0], kernel_size=[box_kernel.shape[2],box_kernel.shape[3]],padding='same', bias=False,groups=x.shape[1])
    with torch.no_grad():
        conv.weight = nn.Parameter(torch.from_numpy(box_kernel))

    #x = torch.from_numpy(np.transpose(x,(0,3,1,2)))
    output = conv(x)
    #output = output.detach().numpy()
    #output = np.transpose(output,(0,2,3,1))
    #output.mean().backward()
    #print(conv.weight.grad)
    #print(output.shape)
    return output


# In[3]:


def guided_filter(x, y, r, eps=1e-2):
    
    x_shape = x.shape
    y_shape = y.shape
    #y_shape = tf.shape(y)
    #print(x_shape,y_shape)
    N = tf_box_filter(torch.ones((1, 1, x_shape[2], x_shape[3]), dtype=x.dtype), r)
    #print(N.shape)
    N_temp = N.detach().numpy()
    N_temp = np.transpose(N_temp,(0,2,3,1))
    #np.save('paper_N.npy',N_temp)

    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x  = tf_box_filter(x * x, r) / N - mean_x * mean_x

    #print(mean_x.shape,mean_y.shape,cov_xy.shape,var_x.shape)
    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    #print(A.shape,b.shape)
    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N

    output = mean_A * x + mean_b

    return output


# In[ ]:




