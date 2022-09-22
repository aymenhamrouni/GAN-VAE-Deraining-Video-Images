#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from keras.models import Sequential
from keras.layers import Input, Concatenate
from keras.layers import Conv2D, Conv2DTranspose, Activation, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import glob
from keras import losses
from sklearn.model_selection import train_test_split
from DataLoader import LoadData, Preprocess
from models import CreateCAE, CreateMMAE
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

#PyTorch lib
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
#Metrics lib
from metrics import calc_psnr, calc_ssim
from keras.callbacks import ModelCheckpoint


# In[9]:


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", type=str, default='CAE')
    parser.add_argument("--mode", type=str, default='raindrops')
    parser.add_argument("--data_dir", type=str, default='../Data')
    parser.add_argument("--traininghardness", type=str, default='light')
    parser.add_argument("--output_dir", type=str, default='./weights')
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--displayfigures", type=bool, default=True)
    args = parser.parse_args()
    return args


# In[10]:


args = get_args()
if args.traininghardness=='light':
    rainstreakDataset=['DID-MDN']
    raindropDataset=['RAINDROP1']
elif args.traininghardness=='meduim':
    rainstreakDataset=['DID-MDN','RAIN800','FU','RAIN100L']
    raindropDataset=['RAINDROP2']
elif  args.traininghardness=='heavy':
    print('WARNING: Training in heavy for rainstreaks/raindrop requires atleast 1TB of RAM')
    rainstreakDataset=['DID-MDN','RAIN100L','RAIN100H','RAINTRAINL','RAINTRAINH','RAIN12600','RAIN1400']
    raindropDataset=['RAINDROP1','RAINDROP2']


# In[11]:


if args.mode=='rainstreaks':
    rained_image, derained_image= LoadData(args.data_dir,rainstreakDataset)
else:
    rained_image, derained_image= LoadData(args.data_dir,raindropDataset)
rained_images, derained_images=Preprocess([rained_image,derained_image],[args.size,args.size])
if args.displayfigures==True:
    SampleRAINYimage=rained_images[600]*255
    SampleNORAINimage=derained_images[600]*255
    plt.imshow(SampleRAINYimage.astype(np.uint8))
    plt.imshow(SampleNORAINimage.astype(np.uint8))
X_train_norain, X_test_norain = train_test_split(derained_images, test_size=0.3, random_state=42)
X_train_rain, X_test_rain = train_test_split(rained_images, test_size=0.3, random_state=42)


# In[26]:


if args.architecture == 'MMAE':

    autoencoder=CreateMMAE([args.size,args.size,3])
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss=losses.mean_squared_error,metrics = ['accuracy'])
    ES = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=10)
    metric='val_loss'
    callbacks_list = [checkpoint,ES]
    checkpoint = ModelCheckpoint(filepath, monitor=metric, verbose=1, save_best_only=True, mode='min')
    Model.fit(autoencoder , [X_train_rain,X_train_norain], [X_train_norain,X_train_norain], validation_data=([X_test_rain,X_test_norain],[X_test_norain,X_test_norain]), epochs=150, shuffle=True, batch_size = 35)

elif args.architecture == 'CAE':
    autoencoder=CreateCAE([args.size,args.size,3])
    autoencoder.summary()
    autoencoder.compile(optimizer='adam', loss=losses.mean_squared_error,metrics = ['accuracy'])
    Model = load_model(args.output_dir+"/"+args.mode+"/firstBatch.hdf5")

    filepath=args.output_dir+"/"+args.mode+"/Epoch-{epoch:02d}-Accuracy-{accuracy:.2f}.hdf5"
    ES = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,patience=100)
    metric='val_accuracy'
    checkpoint = ModelCheckpoint(filepath, monitor=metric, verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint,ES]
    History=Model.fit(X_train_rain, X_train_norain,  validation_data=(X_test_rain, X_test_norain), epochs=150,shuffle=True, callbacks=callbacks_list, batch_size = 35)
    #autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_crossentropy'])
    #autoencoder.save("./weights/raindrops/raindrop.hdf5")


# In[30]:


if args.displayfigures==True:
    plt.plot(History.history['loss'])
    plt.plot(History.history['val_loss'])
    plt.plot(History.history['val_accuracy'])
    plt.plot(History.history['accuracy'])
    plt.title('model loss and accuracy')
    plt.ylabel('loss and accuracy')
    plt.xlabel('epoch')
    plt.legend(['accuracy','val_accuracy','loss',"val_loss"], loc='upper left')
    plt.show()


# In[31]:


print(History.history)


# Train Multi Model Autoencoder

# In[ ]:


autoencoder.save(".weights/rainstreaks/MMAE.hdf5")

