#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import torch as torch
import torch.nn as nn


# In[5]:


def adaptive_instance_norm( content_img , style_img , epsilon = 1e-5) :
    content_var , content_mean = torch.var_mean( content_img , axes=[1, 2], keepdim = True)
    style_var , style_mean = torch.var_mean( style_img , axes=[1, 2], keepdim = True)
    conten_std , style_std = torch.sqrt(content_var+epsilon) , torch.sqrt(style_var+epsilon)
    
    return style_std * (content_img - content_mean) / ( conten_std + style_mean )

