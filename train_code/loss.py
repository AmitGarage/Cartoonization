#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
from torch.autograd import Variable
import numpy as np
import scipy.stats as st
import torchvision.models as models
import copy
import torch.nn as nn
import network
from os.path import exists

# In[3]:


VGG_MEAN = [103.939, 116.779, 123.68]


# In[4]:

device_to_use='cpu'

class vgg19beforefc :

    def __init__(self, vgg19_pt_path=None):
        self.cnn = models.vgg19(pretrained=False).features.to(device_to_use).eval()
        self.previous_weights = torch.load(vgg19_pt_path)
        #self.data_dict = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        #print('Finished loading vgg19.npy')
        self.vgg19_model = self.vgg19_definition()
        #print(self.vgg19_model)
    
    def build_model(self , rgb, include_fc=False) :
        
        rgb_scaled = (rgb+1) * 127.5
        #print(rgb_scaled.shape)
        blue, green, red = torch.split(tensor=rgb_scaled,split_size_or_sections=1,dim=1)
        bgr = torch.cat(tensors=(blue - VGG_MEAN[0],green - VGG_MEAN[1], red - VGG_MEAN[2]),dim=1)
        #print(bgr.shape)
        return self.vgg19_model(bgr)
        
    def vgg19_definition(self) :
        cnn = copy.deepcopy(self.cnn)
        #print(cnn)
        # Initalize model and add layers in sequential order
        model = nn.Sequential()
        # Initalize layer number
        conv_layer = 0
        # Loop to add layer and layer name to model
        for layer in cnn.children() :
            if isinstance(layer ,nn.Conv2d) :
                conv_layer += 1
                layer_name = 'conv_{}'.format(conv_layer)
            elif isinstance(layer ,nn.ReLU) :
                layer_name = 'relu_{}'.format(conv_layer)
                layer = nn.ReLU(inplace = False)
            elif isinstance(layer ,nn.MaxPool2d) :
                layer_name = 'maxPool_{}'.format(conv_layer)
            elif isinstance(layer ,nn.BatchNorm2d) :
                layer_name = 'batchNorm_{}'.format(conv_layer)
            else :
                raise RuntimeError('Unrecognized layer : {}'.format(layer.__class__.__name__))
            
            model.add_module(layer_name , layer)
            #print(model)
         
            #print(conv_layer)
        # Loop from reverse to count no of layer in model excluding any flatten layer
        #for conv_layer in range(len(model)-1,-1,-1) :
        #    if isinstance(model[conv_layer], PicContentLoss) or isinstance(model[conv_layer], PicStyleLoss) :
        #        break
    
        #print(conv_layer)  
        #print(model[conv_layer])
        # Final model over which training will be done
        #model = model[:(conv_layer+1)]
        #print(model)
        
        return model


# In[5]:


def vgg19loss(image_a,image_b) :
    vgg_model = Vgg19('vgg19_no_fc.pt')
    vgg_model_image_a = vgg_model.build_model(image_a)
    vgg_model_image_b = vgg_model.build_model(image_b)
    loss = torch.nn.L1Loss()
    vgg_loss = loss(vgg_model_image_a-vgg_model_image_b)
    h, w, c= vgg_a.get_shape().as_list()[1:]
    vgg_loss = torch.mean(vgg_loss)/(h*w*c)
        
    return vgg_loss
    
def wgan_loss(discriminator, real, fake, patch=True, channel=32, lambda_=2) :
    real_logits = discriminator(real, patch=patch, channel=channel)
    fake_logits = discriminator(fake, patch=patch, channel=channel)

    d_loss_real = - torch.mean(real_logits)
    d_loss_fake = torch.mean(fake_logits)

    d_loss = d_loss_real + d_loss_fake
    g_loss = - d_loss_fake

    """ Gradient Penalty """
    # This is borrowed from https://github.com/kodalinaveen3/DRAGAN/blob/master/DRAGAN.ipynb
    alpha = torch.rand([real.shape[0], 1, 1, 1])
    differences = fake - real # This is different from MAGAN
    interpolates = real + (alpha * differences)
    inter_logit = discriminator(interpolates, channel=channel)
    inter_logit.backward(torch.ones(interpolates.shape))
    gradients = interpolates.grad[0]
    #gradients = tf.gradients(inter_logit, [interpolates])[0]
    slopes = torch.sqrt(torch.sum(torch.square(gradients), axis=[1]))
    gradient_penalty = torch.mean((slopes - 1.) ** 2)
    d_loss += lambda_ * gradient_penalty
 
    return d_loss, g_loss
    
def gan_loss(discriminator, real, fake, scale=1,channel=32, patch=False):
    real_logit = discriminator(real, scale, channel, patch=patch)
    fake_logit = discriminator(fake, scale, channel, patch=patch)

    real_logit = torch.nn.Sigmoid(real_logit)
    fake_logit = torch.nn.Sigmoid(fake_logit)
    
    g_loss_blur = -torch.mean(torch.log(fake_logit)) 
    d_loss_blur = -torch.mean(torch.log(real_logit) + torch.log(1. - fake_logit))

    return d_loss_blur, g_loss_blur



def lsgan_loss(discriminator , real, fake, scale=1, channel=32, patch=False):
    #if discriminator == 'Discriminator_SpectralNorm' :
    #    input_channel = real.shape[1]
    #    Disc_SN = network.Discriminator_SpectralNorm(scale, channel, patch=patch, input_channel=input_channel)
    #print(real.shape)
    real_logit = discriminator(real)
    print(real_logit.shape)
    if exists('/content/paper_disc_gray_real.npy') :
      result_out = real_logit.detach().numpy()
      result_out = np.transpose(result_out,(0,2,3,1))
      np.save('paper_disc_blur_real.npy',result_out)
    else :
      result_out = real_logit.detach().numpy()
      result_out = np.transpose(result_out,(0,2,3,1))
      np.save('paper_disc_gray_real.npy',result_out)
    #    input_channel = fake.shape[1]
    #    Disc_SN = network.Discriminator_SpectralNorm(scale, channel, patch=patch, input_channel=input_channel)
    #print(fake.shape)
    fake_logit = discriminator(fake)
    print(fake_logit.shape)
    if exists('/content/paper_disc_gray_fake.npy') :
      result_out = fake_logit.detach().numpy()
      result_out = np.transpose(result_out,(0,2,3,1))
      np.save('paper_disc_blur_fake.npy',result_out)
    else :
      result_out = fake_logit.detach().numpy()
      result_out = np.transpose(result_out,(0,2,3,1))
      np.save('paper_disc_gray_fake.npy',result_out)

    g_loss = torch.mean((fake_logit - 1)**2)
    d_loss = 0.5*(torch.mean((real_logit - 1)**2) + torch.mean(fake_logit**2))
    
    return d_loss, g_loss

def total_variation_loss(image, k_size=1):
    h, w = list(image.shape)[2:]
    tv_h = torch.mean((image[:, :,k_size:, :] - image[:, :,:h - k_size, :])**2)
    tv_w = torch.mean((image[:, :, :,k_size:] - image[:, :, :,:w - k_size])**2)
    tv_loss = (tv_h + tv_w)/(3*h*w)
    return tv_loss


# In[ ]:




