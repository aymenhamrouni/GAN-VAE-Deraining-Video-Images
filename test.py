#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import keras
import os
import torch
import numpy as np
import cv2
import pafy
from time import time
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import keras
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
#Tools lib
import numpy as np
import cv2
import random
import time
import os
import argparse
from models_custom import *
from skimage.metrics import structural_similarity as ssim
from statistics import mean
from vif_utils import vif, fsim
from CVcheck import ObjectDetection
from Denoise import StreamDenoise
import argparse
from metrics import calc_psnr, calc_ssim

from config import global_config

CFG = global_config.cfg

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Available device:', device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
print('Using CPU..')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, default='images') #it can be video or images
    parser.add_argument("--mode", type=str, default='evaluate') #it can be evaluate or demo
    parser.add_argument("--object_detection_test", type=bool, default=True)
    parser.add_argument("--input", type=str, default='./input')
    parser.add_argument("--arch", type=str, default='CAE') #it can be MMAE, GAN
    parser.add_argument("--rain", type=str, default='rainstreaks') #it can be rainstreaks or raindrops
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument("--weights", type=str, default='./weights')
    parser.add_argument("--enhancer", type=str, default='GAN')# it can be GAN or Autoencoder

    args = parser.parse_args()
    return args


# In[2]:


args = get_args()

print('Testing Denoising Model: ', args.arch )
print('Images are: ', args.rain )
if args.arch=='CAE':
    print('Enhancer is: ', args.enhancer )
    if args.rain=='rainstreaks':
        denoisingModel = load_model(args.weights+"/rainstreaks/DerainAdvanced.hdf5",compile=False)
    else:
        denoisingModel = load_model(args.weights+"/raindrops/post-trained.hdf5",compile=False)
    if args.enhancer=='GAN':
        enhancerModel = load_model(args.weights+"/GAN_enhancer.h5",compile=False)
    else:
        enhancerModel = load_model(args.weights+"/AE_Enhancer.hdf5",compile=False)
elif args.arch=='GAN':
    denoisingModel = Generator().cpu()
    denoisingModel.load_state_dict(torch.load(args.weights+"/raindrops/GAN_generator.pkl"))
    enhancerModel=denoisingModel


# In[3]:




if args.type=='images':
    print('Loading test images...')
    denoise = StreamDenoise(args.input+'/images',args.output_dir)
    denoise('images',denoisingModel,enhancerModel,args.arch)
    print('Image denoising completed!')
    print('Image denoising files are saved in ',args.output_dir+"/images/")
    if args.object_detection_test==True:
        print('Running Computer vision algorithms for detection for images before denoising...')
        check = ObjectDetection(args.input+'/images',args.output_dir+"/precvimages")
        check('images')
        print('Computer vision algorithms for detection completed!')
        print('Resultant images are saved in: ',args.output_dir+"/precvimages")
           
        print('Running Computer vision algorithms for detection for images after denoising...')
        check = ObjectDetection(args.output_dir+'/images',args.output_dir+"/imagesaftercv")
        check('images')
        print('Computer vision algorithms for detection completed!')
        print('Resultant images are saved in: ',args.output_dir+"/imagesaftercv")


elif args.type=='video': 
    print('Treating video...')
    fourcc = cv2.VideoWriter_fourcc(*'mp4a')
    print(fourcc)
    denoise = StreamDenoise(args.input+'/video/input.mp4',args.output_dir)
    denoise('video',denoisingModel,enhancerModel,args.arch)
    print('Video denoising completed! Files are saved in ',args.output_dir+"/videos/result.mp4")
    if args.object_detection_test==True:
        print('Running Computer vision algorithms for detection for videos after  denoising...')
        check = ObjectDetection(args.output_dir+'/video/input.mp4',args.output_dir)
        check('video')
        print('Computer vision algorithms for detection completed!')
        
if args.mode=='evaluate':
    avgpsnr=0
    avgssim=0
    mse = []
    vifr = []
    vifb = []
    vifg = []
    FIM=[]
    
    for i in glob.glob(args.input+"/images/*.png"):
        print('Evaluating images: ',i)
        print(os.path.basename(i))
        label_image_vis_gray  = cv2.resize(
            cv2.imread(i, cv2.IMREAD_COLOR), (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
        )
        output_image_gray = cv2.resize(
            cv2.imread(args.output_dir+"/derained_images/no"+os.path.basename(i), cv2.IMREAD_COLOR), (CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), interpolation=cv2.INTER_LINEAR
        )
        
        psnr = calc_psnr(label_image_vis_gray, output_image_gray)
        ssim = calc_ssim(label_image_vis_gray, output_image_gray)
        avgpsnr+=psnr
        avgssim+=ssim
        print('SSIM: {:.5f}'.format(ssim))
        print('PSNR: {:.5f}'.format(psnr))
        mse.append(((label_image_vis_gray - output_image_gray) ** 2).mean(axis=None))
        vifr.append(vif(label_image_vis_gray[:, :, 0].reshape(CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), output_image_gray[:, :, 0].reshape(CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT)))
        vifg.append(vif(label_image_vis_gray[:, :, 1].reshape(CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), output_image_gray[:, :, 1].reshape(CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT)))
        vifb.append(vif(label_image_vis_gray[:, :, 2].reshape(CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT), output_image_gray[:, :, 2].reshape(CFG.TEST.IMG_WIDTH, CFG.TEST.IMG_HEIGHT)))
        FIM.append(fsim(label_image_vis_gray, output_image_gray))
    print("Average MSE", mean(mse))
    print("Average VIF", mean(vifr + mean(vifg) + mean(vifb)) / 3)
    print("Average FSIM", mean(FIM))   
    print('Average SSIM {:.5f}'.format(avgssim/len(os.listdir(args.input+"/images"))))
    print('Average PSNR {:.5f}'.format(avgpsnr/len(os.listdir(args.input+"/images"))))

    
            


# In[ ]:




