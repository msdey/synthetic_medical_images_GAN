import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
import random
import sys
from PIL import Image
import cv2
import h5py
from sklearn.model_selection import train_test_split
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr


from keras.models import load_model
from keras.engine.topology import Layer
from keras import initializers, constraints
from keras.models import Sequential, Model
from keras.layers import Subtract, Add, Lambda ,AveragePooling2D, Deconvolution2D,  Input, merge, Reshape, Dense, Flatten, Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization, Activation, Conv2DTranspose, Dropout, concatenate, Add, Concatenate
from keras.callbacks import TensorBoard, Callback, EarlyStopping, ModelCheckpoint
import keras.backend as K
from keras.utils import np_utils, conv_utils, Sequence, plot_model
from keras import optimizers
from keras.regularizers import l2
from keras.engine.topology import get_source_inputs
from keras.layers.merge import add, multiply
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D

from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from keras.models import load_model

from collections import deque


noise_shape = (1,1,100)
num_steps = 90000
batch_size = 16
image_shape = None
img_save_dir = './output_files'
save_model = True
image_shape = (256,256,1)

gt_data_dir =  './stare_dataset/ground_truth/*ppm'
img_data_dir = './stare_dataset/images/*ppm'

x_ray_data = './chest_x_ray/images/*png'

if os.path.exists(img_save_dir)== False:
  os.mkdir(img_save_dir)

log_dir = img_save_dir
save_model_dir = img_save_dir

def get_images(image_path):
  files = list(glob.glob(image_path))#sorted(os.listdir(image_path))
  images = []
  for i in range(len(files)):
    #file_name = os.path.join(image_path, files[i])
    img = Image.open(files[i])
    img = img.resize(image_shape[:-1])
    #img = img.convert('RGB')
    img = np.asarray(img).astype('float16')
    img = normalise(img)
    img = img[..., np.newaxis]
    images.append(img)
  images = np.asarray(images)
  return images

def normalise(img_arr):
    #img_arr = (img_arr / 127.5) - 1 # between -1 and 1 normalisation (recommended)
    img_arr = img_arr/255.  # Between 0 and 1
    return img_arr

def denormalise(img_arr):
    #for output
    #img_arr = (img_arr + 1) * 127.5 # converting from [-1, 1] to [0, 255]
    img_arr = img_arr*255. # converting from [0, 1] to [0, 255]
    return img_arr.astype(np.uint8) 


def sample_from_dataset(batch_size, image_shape, data_dir=None, data = None):
    sample_dim = (batch_size,) + image_shape
    sample = np.empty(sample_dim, dtype=np.float32)
    all_data_dirlist = list(glob.glob(data_dir))
    sample_imgs_paths = np.random.choice(all_data_dirlist,batch_size)
    for index,img_filename in enumerate(sample_imgs_paths):
        image = Image.open(img_filename)
        image = image.resize(image_shape[:-1])
        image = image.convert('RGB')
        image = np.asarray(image)
        image = norm_img(image)
        sample[index,...] = image
    return sample


def generate_noise(batch_size, noise_shape):
    return np.random.normal(-1, 1, size=(batch_size,)+noise_shape) # generating noise between -1 and +1


def generate_images(generator, img_save_dir):
    noise = generate_noise(batch_size,noise_shape)
    fake_data_X = generator.predict(noise)
    print("Displaying generated images")
    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(fake_data_X.shape[0],16,replace=False)
    for i in range(16):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = fake_data_X[rand_index, :,:,0]
        fig = plt.imshow(denormalise(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(img_save_dir+str(time.time())+"_generated_image.png",bbox_inches='tight',pad_inches=0)
    plt.show()


def save_img_batch(img_batch,img_save_dir):
    plt.figure(figsize=(4,4))
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(img_batch.shape[0],16,replace=False)

    for i in range(16):
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        image = img_batch[i, :,:,:]
        image = denormalise(image)
        fig = plt.imshow(np.squeeze(image), cmap = 'gray')
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    

    fig = plt.imshow(np.squeeze(image), cmap = 'gray')
    plt.savefig(img_save_dir,bbox_inches='tight',pad_inches=0)
    plt.show()   
