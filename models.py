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
import glob
from keras import losses
from sklearn.model_selection import train_test_split
from DataLoader import LoadData, Preprocess



def CreateCAE(d):
    
    
    input_img = Input(shape=(d[0], d[1], d[2]))

    x = Conv2D(16, (4, 4), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    #encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid',padding = 'same')(x)

    #decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    
    return autoencoder




def CreateMMAE(d):
    
    input_img_1 = Input(shape=(d[0], d[1], d[2]))
    input_img_2 = Input(shape=(d[0], d[1], d[2]))

    x = Conv2D(16, (4, 4), activation='relu', padding='same')(input_img_1)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x_1 = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (4, 4), activation='relu', padding='same')(input_img_2)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    x_2 = MaxPooling2D((2, 2), padding='same')(x)


    x = keras.layers.concatenate([x_1, x_2])
    encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    #encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded_1 = Conv2D(3, (3, 3), activation='sigmoid',padding = 'same')(x)


    x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded_2 = Conv2D(3, (3, 3), activation='sigmoid',padding = 'same')(x)



    #decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(inputs=[input_img_1, input_img_2], outputs=[decoded_1,decoded_2])
    
    return autoencoder


