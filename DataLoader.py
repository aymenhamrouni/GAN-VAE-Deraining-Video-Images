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


def LoadData(Datapath,Datasets):
    coupledDataSet=['DID-MDN','RAIN800','FU']
    totalimages = []
    rained_image = []
    derained_image = []
    for dataset in Datasets:
        if dataset in coupledDataSet and dataset!='FU':
            print("Loading rainstreaks images from: "+Datapath+"/"+dataset)
            for image in glob.glob(Datapath+"/"+dataset+"/**/*.*",recursive=True):
                print('COUPLED DATASET: READING',image)
                totalimages.append(cv2.imread(image))
        if dataset in coupledDataSet and dataset=='FU':        
            print("Loading images from: "+Datapath+"/"+dataset)
            for image in glob.glob(Datapath+"/"+dataset+"/**/*.*",recursive=True):
                print('COUPLED DATASET: READING',image)
                Im=cv2.imread(image)
                col = Im.shape[1]
                lefthalf = Im[:,:col//2,:] 
                righthalf = Im[:,col//2:,:] 
                derained_image.append(lefthalf)
                rained_image.append(righthalf)
                       
    for image in totalimages:
        col = image.shape[1]
        lefthalf = image[:,:col//2,:] 
        righthalf = image[:,col//2:,:] 
        derained_image.append(righthalf)
        rained_image.append(lefthalf)
        
            
            
    for dataset in Datasets:
        if dataset in ['RAIN12600','RAIN1400']:
            print("Loading rainstreaks images from: "+Datapath+"/"+dataset)
            for image in glob.glob(Datapath+"/"+dataset+"/ground_truth/*.*",recursive=True):
                iid=image.split('/')[-1]
                ext=iid[-4:]
                iid=iid[0:-4]
                print('SEPERATE DATASET: READING',Datapath+"/"+dataset+"/ground_truth/"+str(iid)+""+str(ext),' and their rainy versions')
                for i in range(1,15):
                    derained_image.append(cv2.imread(Datapath+"/"+dataset+"/ground_truth/"+str(iid)+""+str(ext)))
                    rained_image.append(cv2.imread(Datapath+"/"+dataset+"/rainy_image/"+str(iid)+"_"+str(i)+str(ext)))
                
                
    
    
    
                
    for dataset in Datasets:
        if dataset in ['RAIN100L','RAIN100H','RAINTRAINL','RAINTRAINH']:
            print("Loading rainstreaks images from: "+Datapath+"/"+dataset)
            for image in glob.glob(Datapath+"/"+dataset+"/ground_truth/*.*",recursive=True):
                iid=image.split('/')[-1]
                ext=iid[-4:]
                iid=iid[7:-4]
                print('SEPERATE DATASET: READING',Datapath+"/"+dataset+"/ground_truth/"+str(iid)+""+str(ext),' and their rainy versions')
                for i in range(1,15):
                    derained_image.append(cv2.imread(Datapath+"/"+dataset+"/ground_truth/norain-"+str(iid)+""+str(ext)))
                    rained_image.append(cv2.imread(Datapath+"/"+dataset+"/rainy_image/rain-"+str(iid)+str(ext)))
                
                
    
    for dataset in Datasets:
        if dataset in ['RAINDROP1']:    
            print("Loading raindrop images from: "+Datapath+"/"+dataset)
            for i in glob.glob("../Data/RAINDROP1/clean/A/*.png"):
                derained_image.append(cv2.imread(i))
                print('Loading ',i,' and their noised versions')
                rainy="../Data/RAINDROP1/rain/A/"+i[26:-10]+"_rain.png"
                rained_image.append(cv2.imread(rainy))

            for i in glob.glob("../Data/RAINDROP1/clean/B/*.png"):
                derained_image.append(cv2.imread(i))
                print('Loading ',i,' and their noised versions')
                rainy="../Data/RAINDROP1/rain/B/"+i[26:-10]+"_rain.png"
                rained_image.append(cv2.imread(rainy))

            for i in glob.glob("../Data/RAINDROP1/clean/C/*.png"):
                derained_image.append(cv2.imread(i))
                print('Loading ',i,' and their noised versions')
                rainy="../Data/RAINDROP1/rain/C/"+i[26:-10]+"_rain.png"
                rained_image.append(cv2.imread(rainy))
    
    
    if dataset in ['RAINDROP2']:    
            print("Loading raindrop images from: "+Datapath+"/"+dataset)
            for i in glob.glob("../Data/RAINDROP2/RAIN_DATASET/ALIGNED_PAIRS/CLEAN/*.png"):
                derained_image.append(cv2.imread(i))
                print('Loading ',i,' and their noised versions')
                rainy="../Data/RAINDROP2/RAIN_DATASET/ALIGNED_PAIRS/CG_DROPLETS"+i[50:]
                rained_image.append(cv2.imread(rainy))
    
    
    
    
    
    
    
    
    print('In total, we now have', len(rained_image), 'pairs of images')
    
    
    
    
    
    
    
    return rained_image, derained_image


def Preprocess(image_Couples,resize_Dimension):
    derained_image=image_Couples[1]
    rained_image=image_Couples[0]
    derain_final = []
    rain_final=[]
    for image in derained_image:
        x = cv2.resize(image,(resize_Dimension[0],resize_Dimension[1]))
        x = x/255
        derain_final.append(x)
        
    for image in rained_image:
        x = cv2.resize(image,(resize_Dimension[0],resize_Dimension[1]))
        x = x/255
        rain_final.append(x)
        
    rain_final = np.asarray(rain_final)
    derain_final = np.asarray(derain_final)

    print('The resultant array vector of one pairs is',rain_final.shape)   
    return rain_final,derain_final
    



                

            
        
   