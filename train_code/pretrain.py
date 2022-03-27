#!/usr/bin/env python
# coding: utf-8

'''
Source code for CVPR 2020 paper
'Learning to Cartoonize Using White-Box Cartoon Representations'
by Xinrui Wang and Jinze yu
'''


import torch

import utils
import os
import numpy as np
import argparse
import network 

from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[14]:


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default = 256, type = int)
    parser.add_argument("--batch_size", default = 16, type = int)     
    parser.add_argument("--total_iter", default = 150, type = int)
    parser.add_argument("--adv_train_lr", default = 2e-4, type = float)
    parser.add_argument("--gpu_fraction", default = 0.5, type = float)
    parser.add_argument("--save_dir", default = '/content/drive/MyDrive/Training_21122021/pretrain', type = str)

    args = parser.parse_args()
    
    return args


# In[ ]:


def initalization( args ):
    Model = network.unet_generator()
    #print(Model)
    #Model.load_state_dict(torch.load('/content/drive/MyDrive/Training_21122021/test_model_weights.pt')['model_state_dict'])
    optim = torch.optim.Adam(Model.parameters(),args.adv_train_lr, betas=(0.5, 0.99))
    #weights = Model.state_dict()['conv1.weight']
    #print(weights.shape)
    #result_out = weights.detach().numpy()
    #result_out = np.transpose(result_out,(2,3,1,0))
    #np.save('paper_pretrain_weights_0.npy',result_out)
    #weights = Model.state_dict()['conv2.weight']
    #print(weights.shape)
    #result_out = weights.detach().numpy()
    #result_out = np.transpose(result_out,(2,3,1,0))
    #np.save('paper_pretrain_weights_1.npy',result_out)
    return Model,optim


# In[2]:


def train(args):
    
    #input_photo = tf.placeholder(tf.float32, [args.batch_size,args.patch_size, args.patch_size, 3])
    #input_superpixel = tf.placeholder(tf.float32, [args.batch_size,args.patch_size, args.patch_size, 3])
    #input_cartoon = tf.placeholder(tf.float32, [args.batch_size,args.patch_size, args.patch_size, 3])
    
    #output = network.unet_generator(input_photo)
    #output = guided_filter(input_photo, output, r=1)

    
    #blur_fake = guided_filter(output, output, r=5, eps=2e-1)
    #blur_cartoon = guided_filter(input_cartoon, input_cartoon, r=5, eps=2e-1)

    #gray_fake, gray_cartoon = utils.color_shift(output, input_cartoon)
    
    #d_loss_gray, g_loss_gray = loss.lsgan_loss(network.disc_sn, gray_cartoon, gray_fake,scale=1, patch=True, name='disc_gray')
    #d_loss_blur, g_loss_blur = loss.lsgan_loss(network.disc_sn, blur_cartoon, blur_fake,scale=1, patch=True, name='disc_blur')


    #vgg_model = loss.Vgg19('vgg19_no_fc.npy')
    #vgg_photo = vgg_model.build_conv4_4(input_photo)
    #vgg_output = vgg_model.build_conv4_4(output)
    #vgg_superpixel = vgg_model.build_conv4_4(input_superpixel)
    #h, w, c = vgg_photo.get_shape().as_list()[1:]
    
    #photo_loss = tf.reduce_mean(tf.losses.absolute_difference(vgg_photo, vgg_output))/(h*w*c)
    #superpixel_loss = tf.reduce_mean(tf.losses.absolute_difference(vgg_superpixel, vgg_output))/(h*w*c)
    #recon_loss = photo_loss + superpixel_loss
    #tv_loss = loss.total_variation_loss(output)
    
    #g_loss_total = 1e4*tv_loss + 1e-1*g_loss_blur + g_loss_gray + 2e2*recon_loss
    #d_loss_total = d_loss_blur + d_loss_gray

    #all_vars = tf.trainable_variables()
    #gene_vars = [var for var in all_vars if 'gene' in var.name]
    #disc_vars = [var for var in all_vars if 'disc' in var.name] 
    
    
    #tf.summary.scalar('tv_loss', tv_loss)
    #tf.summary.scalar('photo_loss', photo_loss)
    #tf.summary.scalar('superpixel_loss', superpixel_loss)
    #tf.summary.scalar('recon_loss', recon_loss)
    #tf.summary.scalar('d_loss_gray', d_loss_gray)
    #tf.summary.scalar('g_loss_gray', g_loss_gray)
    #tf.summary.scalar('d_loss_blur', d_loss_blur)
    #tf.summary.scalar('g_loss_blur', g_loss_blur)
    #tf.summary.scalar('d_loss_total', d_loss_total)
    #tf.summary.scalar('g_loss_total', g_loss_total)
      
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
        
        #g_optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99).minimize(g_loss_total, var_list=gene_vars)
        
        #d_optim = tf.train.AdamOptimizer(args.adv_train_lr, beta1=0.5, beta2=0.99).minimize(d_loss_total, var_list=disc_vars)
           
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    '''
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_fraction)
    #sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    
    #train_writer = tf.summary.FileWriter(args.save_dir+'/train_log')
    #summary_op = tf.summary.merge_all()
    #saver = tf.train.Saver(var_list=gene_vars, max_to_keep=20)
   
    #with tf.device('/device:GPU:0'):
    
    #device_to_use = ('cuda' if torch.cuda.is_available() else 'cpu')
    device_to_use = 'cpu'
    
    #latest_checkpoint = torch.load('/content/drive/MyDrive/model_weights.pt',map_location=torch.device(device_to_use))

    Model,optim = initalization( args )

    face_photo_dir = '/content/drive/MyDrive/Training_21122021/dataset/photo_face'
    if os.path.isdir("/content/drive/MyDrive/Training_21122021/dataset/photo_face/.ipynb_checkpoints") :
        os.rmdir("/content/drive/MyDrive/Training_21122021/dataset/photo_face/.ipynb_checkpoints")
    face_photo_list = utils.load_image_list(face_photo_dir)
    scenery_photo_dir = '/content/drive/MyDrive/Training_21122021/dataset/photo_scenery'
    if os.path.isdir("/content/drive/MyDrive/Training_21122021/dataset/photo_scenery/.ipynb_checkpoints") :
        os.rmdir("/content/drive/MyDrive/Training_21122021/dataset/photo_scenery/.ipynb_checkpoints")
    scenery_photo_list = utils.load_image_list(scenery_photo_dir)
    
    for total_iter in tqdm(range(args.total_iter)):

        if np.mod(total_iter, 5) == 0: 
            photo_batch = utils.next_batch(face_photo_list, args.batch_size)
        else:
            photo_batch = utils.next_batch(scenery_photo_list, args.batch_size)
        
        #inter_out = sess.run(output, feed_dict={input_photo: photo_batch,input_superpixel: photo_batch,input_cartoon: cartoon_batch})
        
        optim.zero_grad()
        
        photo_batch = torch.from_numpy(np.transpose(photo_batch,(0,3,1,2)))
        output = Model(photo_batch)

        #print(photo_batch.shape,output.shape)
        result_out = output.detach().numpy()
        result_out = np.transpose(result_out,(0,2,3,1))
        np.save('paper_pretrain_out.npy',result_out)
        recon_loss = (torch.nn.functional.l1_loss(photo_batch, output)).mean()
        
        recon_loss.backward()

        r_loss = recon_loss

        torch.save({'state_dict': Model.state_dict(), 'optimizer': optim.state_dict(),}, '/content/drive/MyDrive/Training_21122021/pretrain/pretrain_model_weights.pt')
            
        if np.mod(total_iter+1, 50) == 0:

                print('pretrain, iter: {}, recon_loss: {}'.format(total_iter, r_loss))
                torch.save({'epoch' : total_iter ,'state_dict': Model.state_dict(), 'optimizer': optim.state_dict(),}, '/content/drive/MyDrive/Training_21122021/pretrain/pretrain_epoch_model_weights.pt')
                if np.mod(total_iter+1, 500 ) == 0:
                    #saver.save(sess, args.save_dir+'/saved_models/model',write_meta_graph=False, global_step=total_iter)
                     
                    photo_face = utils.next_batch(face_photo_list, args.batch_size)
                    photo_scenery = utils.next_batch(scenery_photo_list, args.batch_size)

                    #result_face = sess.run(output, feed_dict={input_photo: photo_face,input_superpixel: photo_face,input_cartoon: cartoon_face})
                    
                    photo_face = torch.from_numpy(np.transpose(photo_face,(0,3,1,2)))
                    output = Model(photo_face)
                    result_face = output
                    
                    #result_scenery = sess.run(output, feed_dict={input_photo: photo_scenery,input_superpixel: photo_scenery,input_cartoon: cartoon_scenery})

                    photo_scenery = torch.from_numpy(np.transpose(photo_scenery,(0,3,1,2)))
                    output = Model(photo_scenery)
                    result_scenery = output
                    result_face = result_face.detach().numpy()
                    result_face = np.transpose(result_face,(0,2,3,1))
                    photo_face = photo_face.detach().numpy()
                    photo_face = np.transpose(photo_face,(0,2,3,1))
                    result_scenery = result_scenery.detach().numpy()
                    result_scenery = np.transpose(result_scenery,(0,2,3,1))
                    photo_scenery = photo_scenery.detach().numpy()
                    photo_scenery = np.transpose(photo_scenery,(0,2,3,1))
                    #total_iter=1
                    utils.write_batch_image(result_face, args.save_dir+'/images',str(total_iter)+'_face_result.jpg', 1)
                    utils.write_batch_image(photo_face, args.save_dir+'/images',str(total_iter)+'_face_photo.jpg', 1)

                    utils.write_batch_image(result_scenery, args.save_dir+'/images',str(total_iter)+'_scenery_result.jpg', 1)
                    utils.write_batch_image(photo_scenery, args.save_dir+'/images',str(total_iter)+'_scenery_photo.jpg', 1)


# In[7]:


if __name__ == '__main__':
    
    args = arg_parser()
    train(args)
