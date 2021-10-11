#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np


# In[2]:


def tf_box_filter(x, r):
    ch = x.get_shape().as_list()[-1]
    weight = 1/((2*r+1)**2)
    #box_kernel = weight*np.ones((2*r+1, 2*r+1, ch, 1))
    box_kernel = weight*np.ones((1, ch, 2*r+1, 2*r+1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    #output = tf.nn.depthwise_conv2d(x, box_kernel, [1, 1, 1, 1], 'SAME')
    conv = nn.Conv2d(x.shape[1],box_kernel.shape[0], kernel_size=[box_kernel.shape[2],box_kernel.shape[3]],padding=0, groups=x.shape[1])
    with torch.no_grad():
        conv.weight = nn.Parameter(torch.from_numpy(box_kernel))

    output = conv(torch.from_numpy(x))
    output.mean().backward()
    #print(conv.weight.grad)
    return output


# In[3]:


def guided_filter(x, y, r, eps=1e-2):
    
    x_shape = x.shape()
    #y_shape = tf.shape(y)

    N = tf_box_filter(tf.ones((1, 1, x_shape[1], x_shape[2]), dtype=x.dtype), r)

    mean_x = tf_box_filter(x, r) / N
    mean_y = tf_box_filter(y, r) / N
    cov_xy = tf_box_filter(x * y, r) / N - mean_x * mean_y
    var_x  = tf_box_filter(x * x, r) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = tf_box_filter(A, r) / N
    mean_b = tf_box_filter(b, r) / N

    output = mean_A * x + mean_b

    return output


# In[ ]:




