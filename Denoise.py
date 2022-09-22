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
import glob
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





class StreamDenoise:


    def __init__(self, path, out):
      
        self.out = out
        self.device = 'cpu' 
        self.path=path

    def get_video(self):

        #play = pafy.new(self._URL).streams[-1]
        #assert play is not None
        return cv2.VideoCapture(self.path)
        #return cv2.VideoCapture(play.url)
        # return live feed stream
        #return stream = cv2.VideoCapture(0)

    def get_image(self):
        images = []
        names=[]
        for i in glob.glob(self.path+"/*.png"):
            print('Loading images: ',i)
            image=cv2.imread(i)
            images.append(image)
            names.append(os.path.basename(i))
            
        images = np.asarray(images)
        return images,names

    def predict(self,image,model):
        image = np.array(image, dtype='float32')/255.
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :, :, :]
        image = torch.from_numpy(image)
        image = Variable(image).cpu()

        out = model(image)[-1]

        out = out.cpu().data
        out = out.numpy()
        out = out.transpose((0, 2, 3, 1))
        out = out[0, :, :, :]*255.

        return out


    def align_to_four(self,img):
        #print ('before alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
        #align to four
        a_row = int(img.shape[0]/4)*4
        a_col = int(img.shape[1]/4)*4
        img = img[0:a_row, 0:a_col]
        #print ('after alignment, row = %d, col = %d'%(img.shape[0], img.shape[1]))
        return img

    
    def __call__(self,typeOfinput,model,enhancer,modeltype):

        if typeOfinput=='video':
            player = self.get_video()
            x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
            y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
            four_cc = cv2.VideoWriter_fourcc(*'MPEG')
            out = cv2.VideoWriter(self.out+"/videos/result.mp4", four_cc, 20, (384, 384))
            while True:
                start_time = time()
                ret, frame = player.read()
                if ret:                  
                    if modeltype=='CAE':
                        x = cv2.resize(frame, (512, 512))
                        x = x / 255
                        x=np.asarray(x)
                        frame=model.predict(x.reshape(1, 512, 512, 3))
                        frame = frame.squeeze()
                        frame = cv2.resize(frame, (96, 96))
                        frame = enhancer.predict(np.expand_dims(frame, axis=0))[0]
                        frame = (((frame + 1) / 2.) * 255).astype(np.uint8)
                    else:
                        frame = self.align_to_four(frame)
                        frame = self.predict(frame,model)
                    end_time = time()
                    print('Processing real-time frames to denoise them')
                    fps = 1/np.round(end_time - start_time, 3)
                    out.write(frame)
                else:
                    break
            out.release()
        else:
            images,names=self.get_image()
            i=0
            for image in images:
                print ('Processing image: %s'%(names[i]))
                if modeltype=='CAE':
                    image = cv2.resize(image, (512, 512))
                    image = image / 255
                    frame=model.predict(image.reshape(1, 512, 512, 3))
                    frame = frame.squeeze()
                    frame = cv2.resize(frame, (96, 96))
                    frame = enhancer.predict(np.expand_dims(frame, axis=0))[0]
                    frame = (((frame + 1) / 2.) * 255).astype(np.uint8)
                    cv2.imwrite(self.out+"/images/"+names[i], frame)
                    i=i+1
                else:
                    frame = self.align_to_four(image)
                    frame = self.predict(frame,model)
                    cv2.imwrite(self.out+"/images/"+names[i], frame)
                    i=i+1
                    
                
