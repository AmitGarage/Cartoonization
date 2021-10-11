#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.ndimage import filters
from skimage import segmentation, color
from joblib import Parallel, delayed
from selective_search.util import switch_color_space
from selective_search.structure import HierarchicalGrouping

import os
import cv2
import numpy as np
import scipy.stats as st
import torch


# In[2]:


def color_shift(image1, image2, mode='uniform'):
    b1, g1, r1 = torch.split(image1, split_size_or_sections=3, dim=3)
    b2, g2, r2 = torch.split(image2, split_size_or_sections=3, dim=3)
    if mode == 'normal':
        b_weight = torch.normal(size=[1], mean=0.114, std=0.1)
        g_weight = torch.normal(size=[1], mean=0.587, std=0.1)
        r_weight = torch.normal(size=[1], mean=0.299, std=0.1)
    elif mode == 'uniform':
        b_weight = (torch.rand(1)*(0.214-0.014))+0.014
        g_weight = (torch.rand(1)*(0.687-0.487))+0.487
        r_weight = (torch.rand(1)*(0.399-0.199))+0.199
    output1 = (b_weight*b1+g_weight*g1+r_weight*r1)/(b_weight+g_weight+r_weight)
    output2 = (b_weight*b2+g_weight*g2+r_weight*r2)/(b_weight+g_weight+r_weight)
    return output1, output2


# In[3]:


def label2rgb(label_field, image, kind='mix', bg_label=-1, bg_color=(0, 0, 0)):

    #std_list = list()
    out = np.zeros_like(image)
    labels = np.unique(label_field)
    bg = (labels == bg_label)
    if bg.any():
        labels = labels[labels != bg_label]
        mask = (label_field == bg_label).nonzero()
        out[mask] = bg_color
    for label in labels:
        mask = (label_field == label).nonzero()
        #std = np.std(image[mask])
        #std_list.append(std)
        if kind == 'avg':
            color = image[mask].mean(axis=0)
        elif kind == 'median':
            color = np.median(image[mask], axis=0)
        elif kind == 'mix':
            std = np.std(image[mask])
            if std < 20:
                color = image[mask].mean(axis=0)
            elif 20 < std < 40:
                mean = image[mask].mean(axis=0)
                median = np.median(image[mask], axis=0)
                color = 0.5*mean + 0.5*median
            elif 40 < std:
                color = image[mask].median(axis=0)
        out[mask] = color
    return out


# In[4]:


def color_ss_map(image, seg_num=200, power=1, 
                 color_space='Lab', k=10, sim_strategy='CTSF'):
    
    img_seg = segmentation.felzenszwalb(image, scale=k, sigma=0.8, min_size=100)
    img_cvtcolor = label2rgb(img_seg, image, kind='mix')
    img_cvtcolor = switch_color_space(img_cvtcolor, color_space)
    S = HierarchicalGrouping(img_cvtcolor, img_seg, sim_strategy)
    S.build_regions()
    S.build_region_pairs()

    # Start hierarchical grouping
    
    while S.num_regions() > seg_num:
        
        i,j = S.get_highest_similarity()
        S.merge_region(i,j)
        S.remove_similarities(i,j)
        S.calculate_similarity_for_new_region()
    
    image = label2rgb(S.img_seg, image, kind='mix')
    image = (image+1)/2
    image = image**power
    image = image/np.max(image)
    image = image*2 - 1
    
    return image


# In[5]:


def selective_adacolor(batch_image, seg_num=200, power=1):
    num_job = np.shape(batch_image)[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(color_ss_map)\(image, seg_num, power) for image in batch_image)
    return np.array(batch_out)


# In[6]:


def simple_superpixel(batch_image, seg_num=200):
    
    def process_slic(image):
        seg_label = segmentation.slic(image, n_segments=seg_num, sigma=1,compactness=10, convert2lab=True)
        image = color.label2rgb(seg_label, image, kind='mix')
        return image
    
    num_job = np.shape(batch_image)[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(process_slic)\(image) for image in batch_image)
    return np.array(batch_out)


# In[7]:


def load_image_list(data_dir):
    name_list = list()
    for name in os.listdir(data_dir):
        name_list.append(os.path.join(data_dir, name))
    name_list.sort()
    return name_list


# In[8]:


def next_batch(filename_list, batch_size):
    idx = np.arange(0 , len(filename_list))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    batch_data = []
    for i in range(batch_size):
        image = cv2.imread(filename_list[idx[i]])
        image = image.astype(np.float32)/127.5 - 1
        #image = image.astype(np.float32)/255.0
        batch_data.append(image)
            
    return np.asarray(batch_data)


# In[9]:


def write_batch_image(image, save_dir, name, n):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fused_dir = os.path.join(save_dir, name)
    fused_image = [0] * n
    for i in range(n):
        fused_image[i] = []
        for j in range(n):
            k = i * n + j
            image[k] = (image[k]+1) * 127.5
            #image[k] = image[k] - np.min(image[k])
            #image[k] = image[k]/np.max(image[k])
            #image[k] = image[k] * 255.0
            fused_image[i].append(image[k])
        fused_image[i] = np.hstack(fused_image[i])
    fused_image = np.vstack(fused_image)
    cv2.imwrite(fused_dir, fused_image.astype(np.uint8))


# In[10]:


if __name__ == '__main__':
    pass


# In[ ]:




