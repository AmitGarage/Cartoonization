#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import loss

from tqdm import tqdm
from guided_filter import guided_filter
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[14]:


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default = 256, type = int)
    parser.add_argument("--batch_size", default = 1, type = int)     
    parser.add_argument("--total_iter", default = 1, type = int)
    parser.add_argument("--adv_train_lr", default = 2e-4, type = float)
    parser.add_argument("--gpu_fraction", default = 0.5, type = float)
    parser.add_argument("--save_dir", default = '/content/drive/MyDrive/Training_21122021/train_cartoon', type = str)
    parser.add_argument("--use_enhance", default = False)

    args = parser.parse_args()
    
    return args


# In[ ]:


def initalization(path_to_previous_weight):
    vgg_model = loss.vgg19beforefc(path_to_previous_weight)
    #print(vgg_model.vgg19_model)
    g_optim = torch.optim.Adam(vgg_model.vgg19_model.parameters(),args.adv_train_lr, betas=(0.5,0.99))

    Disc_SN_1 = network.Discriminator_SpectralNorm(scale=1, channel=32, patch=True, input_channel=1)
    #print(Disc_SN_1)
    #print(Disc_SN_1.state_dict().keys())
    Disc_SN_1.load_state_dict(torch.load('/content/drive/MyDrive/Training_21122021/disc_sn_1_weights.pt')['model_state_dict'])

    d_optim_1 = torch.optim.Adam(Disc_SN_1.parameters(),args.adv_train_lr, betas=(0.5,0.99))

    Disc_SN_3 = network.Discriminator_SpectralNorm(scale=1, channel=32, patch=True, input_channel=3)
    #print(Disc_SN_3.state_dict().keys())
    Disc_SN_3.load_state_dict(torch.load('/content/drive/MyDrive/Training_21122021/disc_sn_3_weights.pt')['model_state_dict'])

    d_optim_3 = torch.optim.Adam(Disc_SN_3.parameters(),args.adv_train_lr, betas=(0.5,0.99))
    
    return vgg_model,g_optim,[Disc_SN_1,Disc_SN_3],[d_optim_1,d_optim_3]


# In[2]:


def train(args,vgg_model,g_optim,Disc_SN,d_optim):
    
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
    
    latest_checkpoint = torch.load('/content/drive/MyDrive/Training_21122021/test_model_weights_0.pt',map_location=torch.device(device_to_use))

    if os.path.isdir("/content/drive/MyDrive/Training_21122021/dataset/photo_face/.ipynb_checkpoints") :
        os.rmdir("/content/drive/MyDrive/Training_21122021/dataset/photo_face/.ipynb_checkpoints")
    if os.path.isdir("/content/drive/MyDrive/Training_21122021/dataset/cartoon_scenery/.ipynb_checkpoints") :
        os.rmdir("/content/drive/MyDrive/Training_21122021/dataset/cartoon_scenery/.ipynb_checkpoints")
    if os.path.isdir("/content/drive/MyDrive/Training_21122021/dataset/cartoon_face/.ipynb_checkpoints") :
        os.rmdir("/content/drive/MyDrive/Training_21122021/dataset/cartoon_face/.ipynb_checkpoints")
    if os.path.isdir("/content/drive/MyDrive/Training_21122021/dataset/photo_scenery/.ipynb_checkpoints") :
        os.rmdir("/content/drive/MyDrive/Training_21122021/dataset/photo_scenery/.ipynb_checkpoints")

    face_photo_dir = '/content/drive/MyDrive/Training_21122021/dataset/photo_face'
    face_photo_list = utils.load_image_list(face_photo_dir)
    scenery_photo_dir = '/content/drive/MyDrive/Training_21122021/dataset/photo_scenery'
    scenery_photo_list = utils.load_image_list(scenery_photo_dir)

    face_cartoon_dir = '/content/drive/MyDrive/Training_21122021/dataset/cartoon_face'
    face_cartoon_list = utils.load_image_list(face_cartoon_dir)
    scenery_cartoon_dir = '/content/drive/MyDrive/Training_21122021/dataset/cartoon_scenery'
    scenery_cartoon_list = utils.load_image_list(scenery_cartoon_dir)

    model_network = network.unet_generator()
    #print(model_network)
    model_network.load_state_dict(torch.load('/content/drive/MyDrive/Training_21122021/test_model_weights_0.pt')['model_state_dict'])
    model_network.eval()
    
    writer = SummaryWriter()
    #d_l_gray = 0
    #g_l_gray = 0
    #d_l_blur = 0
    #g_l_blur = 0
    for total_iter in tqdm(range(args.total_iter)):

        if np.mod(total_iter, 5) == 0: 
            photo_batch = utils.next_batch(face_photo_list, args.batch_size)
            cartoon_batch = utils.next_batch(face_cartoon_list, args.batch_size)
        else:
            photo_batch = utils.next_batch(scenery_photo_list, args.batch_size)
            cartoon_batch = utils.next_batch(scenery_cartoon_list, args.batch_size)
        
        #inter_out = sess.run(output, feed_dict={input_photo: photo_batch,input_superpixel: photo_batch,input_cartoon: cartoon_batch})
        
        Disc_SN[0].zero_grad()
        Disc_SN[1].zero_grad()
        
        #print(photo_batch.shape)
        photo_batch = torch.from_numpy(np.transpose(photo_batch,(0,3,1,2)))
        cartoon_batch = torch.from_numpy(np.transpose(cartoon_batch,(0,3,1,2)))
        #print(photo_batch.shape)
        #var_1 = model_network.state_dict()['conv1.weight']
        #print(var_1.shape)
        #result_out = var_1.detach().numpy()
        #result_out = np.transpose(result_out,(2,3,1,0))
        #np.save('paper_unet_1.npy',result_out)
        output = model_network(photo_batch)
        result_out = output.detach().numpy()
        result_out = np.transpose(result_out,(0,2,3,1))
        np.save('paper_output.npy',result_out)
        #print(photo_batch.shape,output.shape)
        output = guided_filter(photo_batch, output, r=1)
        
        result_out = output.detach().numpy()
        result_out = np.transpose(result_out,(0,2,3,1))
        np.save('paper_inter_out.npy',result_out)
        #inter_out = output
        
        blur_fake = guided_filter(output, output, r=5, eps=2e-1)
        result_out = blur_fake.detach().numpy()
        result_out = np.transpose(result_out,(0,2,3,1))
        np.save('paper_blur_fake.npy',result_out)
        blur_cartoon = guided_filter(cartoon_batch, cartoon_batch, r=5, eps=2e-1)
        result_out = blur_cartoon.detach().numpy()
        result_out = np.transpose(result_out,(0,2,3,1))
        np.save('paper_blur_cartoon.npy',result_out)

        gray_fake, gray_cartoon = utils.color_shift(output, cartoon_batch)
        result_out = gray_fake.detach().numpy()
        result_out = np.transpose(result_out,(0,2,3,1))
        np.save('paper_gray_fake.npy',result_out)
        result_out = gray_cartoon.detach().numpy()
        result_out = np.transpose(result_out,(0,2,3,1))
        np.save('paper_gray_cartoon.npy',result_out)

        '''
        adaptive coloring has to be applied with the clip_by_value 
        in the last layer of generator network, which is not very stable.
        to stabiliy reproduce our results, please use power=1.0
        and comment the clip_by_value function in the network.py first
        If this works, then try to use adaptive color with clip_by_value.
        '''
            
        if args.use_enhance:
            superpixel_batch = utils.selective_adacolor(output, power=1.2)
        else:
            superpixel_batch = utils.simple_superpixel(output, seg_num=200)
        superpixel_batch = torch.from_numpy(np.transpose(superpixel_batch,(0,3,1,2)))
        result_out = superpixel_batch.detach().numpy()
        result_out = np.transpose(result_out,(0,2,3,1))
        np.save('paper_superpixel_batch.npy',result_out)
        #_, g_loss, r_loss = sess.run([g_optim, g_loss_total, recon_loss],feed_dict={input_photo: photo_batch,input_superpixel: superpixel_batch,input_cartoon: cartoon_batch})
        

        #_, d_loss, train_info = sess.run([d_optim, d_loss_total, summary_op],feed_dict={input_photo: photo_batch,input_superpixel: superpixel_batch,input_cartoon: cartoon_batch})
        
        d_loss_gray, g_loss_gray = loss.lsgan_loss(Disc_SN[0], gray_cartoon, gray_fake, scale=1, patch=True)
        d_loss_blur, g_loss_blur = loss.lsgan_loss(Disc_SN[1], blur_cartoon, blur_fake, scale=1, patch=True)
        
        #d_l_gray, g_l_gray , d_l_blur, g_l_blur = d_loss_gray, g_loss_gray , d_loss_blur, g_loss_blur

        d_loss_total = d_loss_blur + d_loss_gray
        d_loss = d_loss_total
        print(d_loss_gray, g_loss_gray, d_loss_blur, g_loss_blur)
        print("D_loss - "+str(d_loss))
        d_loss.backward(retain_graph=True)

        #print(vgg_model.vgg19_model)
        vgg_model.vgg19_model.zero_grad()
        #print(photo_batch.shape)
        #output = model_network(photo_batch)
        #print("Output")
        #print(photo_batch.shape,output.shape)
        #output = guided_filter(photo_batch, output, r=1)
        #result_out = output.detach().numpy()
        #result_out = np.transpose(result_out,(0,2,3,1))
        #np.save('paper_inter_out_1.npy',result_out)
        #inter_out = output
        
        #blur_fake = guided_filter(output, output, r=5, eps=2e-1)
        #blur_cartoon = guided_filter(cartoon_batch, cartoon_batch, r=5, eps=2e-1)

        #gray_fake, gray_cartoon = utils.color_shift(output, cartoon_batch)
        #result_out = gray_fake.detach().numpy()
        #result_out = np.transpose(result_out,(0,2,3,1))
        #np.save('paper_gray_fake_1.npy',result_out)
        #result_out = gray_cartoon.detach().numpy()
        #result_out = np.transpose(result_out,(0,2,3,1))
        #np.save('paper_gray_cartoon_1.npy',result_out)
        #d_loss_gray, g_loss_gray = loss.lsgan_loss(Disc_SN[0], gray_cartoon, gray_fake, scale=1, patch=True)
        #d_loss_blur, g_loss_blur = loss.lsgan_loss(Disc_SN[1], blur_cartoon, blur_fake, scale=1, patch=True)
        #print(d_l_gray, g_l_gray , d_l_blur, g_l_blur)
        print(d_loss_gray, g_loss_gray , d_loss_blur, g_loss_blur)
        #d_loss_gray, g_loss_gray , d_loss_blur, g_loss_blur = d_l_gray, g_l_gray , d_l_blur, g_l_blur
        vgg_photo = vgg_model.build_model(photo_batch)
        vgg_output = vgg_model.build_model(output)
        vgg_superpixel = vgg_model.build_model(superpixel_batch)
        c, h, w = list(vgg_photo.shape)[1:]
        #print(c,h,w)
        
        main_loss = torch.nn.L1Loss()
        photo_loss = torch.mean(main_loss(vgg_photo, vgg_output))/(h*w*c)
        superpixel_loss = torch.mean(main_loss(vgg_superpixel, vgg_output))/(h*w*c)
        recon_loss = photo_loss + superpixel_loss
        tv_loss = loss.total_variation_loss(output)
        print(photo_loss , superpixel_loss ,d_loss_gray, g_loss_gray, d_loss_blur, g_loss_blur)
        g_loss_total = 1e4*tv_loss + 1e-1*g_loss_blur + g_loss_gray + 2e2*recon_loss
        
        g_loss = g_loss_total
        r_loss = recon_loss
        print("G_loss - "+str(g_loss))
        print("R_loss - "+str(r_loss))
        g_loss.backward()
        g_optim.step()
        d_optim[0].step()
        d_optim[1].step()
        
        writer.add_scalar('tv_loss', tv_loss, total_iter)
        writer.add_scalar('photo_loss', photo_loss, total_iter)
        writer.add_scalar('superpixel_loss', superpixel_loss, total_iter)
        writer.add_scalar('recon_loss', recon_loss, total_iter)
        writer.add_scalar('d_loss_gray', d_loss_gray, total_iter)
        writer.add_scalar('g_loss_gray', g_loss_gray, total_iter)
        writer.add_scalar('d_loss_blur', d_loss_blur, total_iter)
        writer.add_scalar('g_loss_blur', g_loss_blur, total_iter)
        writer.add_scalar('d_loss_total', d_loss_total, total_iter)
        writer.add_scalar('g_loss_total', g_loss_total, total_iter)
        
        if np.mod(total_iter+1, 50) == 0:
              
            print('Iter: {}, d_loss: {}, g_loss: {}, recon_loss: {}'.                        format(total_iter, d_loss, g_loss, r_loss))
            if np.mod(total_iter+1, 500 ) == 0:
                saver.save(sess, args.save_dir+'/saved_models/model',write_meta_graph=False, global_step=total_iter)
                     
                photo_face = utils.next_batch(face_photo_list, args.batch_size)
                cartoon_face = utils.next_batch(face_cartoon_list, args.batch_size)
                photo_scenery = utils.next_batch(scenery_photo_list, args.batch_size)
                cartoon_scenery = utils.next_batch(scenery_cartoon_list, args.batch_size)

                #result_face = sess.run(output, feed_dict={input_photo: photo_face,input_superpixel: photo_face,input_cartoon: cartoon_face})
                    
                output = network.unet_generator(photo_face)
                output = guided_filter(photo_face, output, r=1)
                result_face = output
                    
                #result_scenery = sess.run(output, feed_dict={input_photo: photo_scenery,input_superpixel: photo_scenery,input_cartoon: cartoon_scenery})

                output = network.unet_generator(photo_scenery)
                output = guided_filter(photo_scenery, output, r=1)
                result_face = output
                    
                utils.write_batch_image(result_face, args.save_dir+'/images',str(total_iter)+'_face_result.jpg', 4)
                utils.write_batch_image(photo_face, args.save_dir+'/images',str(total_iter)+'_face_photo.jpg', 4)

                utils.write_batch_image(result_scenery, args.save_dir+'/images',str(total_iter)+'_scenery_result.jpg', 4)
                utils.write_batch_image(photo_scenery, args.save_dir+'/images',str(total_iter)+'_scenery_photo.jpg', 4)

# In[7]:


if __name__ == '__main__':
    
    args = arg_parser()
    vgg_model,g_optim,Disc_SN,d_optim = initalization('/content/drive/MyDrive/Training_21122021/test_model_weights_0.pt')
    train(args,vgg_model,g_optim,Disc_SN,d_optim)


# In[ ]:




