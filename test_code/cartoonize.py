#!/usr/bin/env python
# coding: utf-8

import os
import cv2
import numpy as np
import torch 
import network
import guided_filter
from tqdm import tqdm



def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image



def cartoonize(load_folder, save_folder, model_path):
    #input_photo = tf.placeholder(tf.float32, [1, None, None, 3])

    #all_vars = tf.trainable_variables()
    #gene_vars = [var for var in all_vars if 'generator' in var.name]
    #saver = tf.train.Saver(var_list=gene_vars)
    
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)

    #sess.run(tf.global_variables_initializer())
    #saver.restore(sess, tf.train.latest_checkpoint(model_path))
    name_list = os.listdir(load_folder)
    model_network = network.unet_generator()
    #print(model_network)
    model_network.load_state_dict(torch.load(model_path)['model_state_dict'])
    model_network.eval()
    for name in tqdm(name_list):
        try:
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)
            image = cv2.imread(load_path)
            image = resize_crop(image)
            batch_image = image.astype(np.float32)/127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            batch_image = torch.from_numpy(np.transpose(batch_image,(0,3,1,2)))
            #output = sess.run(final_out, feed_dict={input_photo: batch_image})
            #print(model_network(torch.from_numpy(batch_image)))
            network_out = model_network(batch_image)
            final_out = guided_filter.guided_filter(batch_image, network_out, r=1, eps=5e-3)
            final_out = final_out.detach().numpy()
            final_out = np.transpose(final_out,(0,2,3,1))
            #np.save('paper_Filter_output.npy',final_out)
            #print(type(final_out),final_out.dtype)
            output = (np.squeeze(final_out)+1)*127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, output)
        except:
            print('cartoonize {} failed'.format(load_path))


if __name__ == '__main__':
    model_path = 'py_saved_model/test_model_weights.pt'
    load_folder = 'test_images'
    save_folder = 'cartoonized_images'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    cartoonize(load_folder, save_folder, model_path)
