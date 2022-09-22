import torch
import numpy as np
import cv2
import pafy
from time import time
import glob
import os

from imageai.Detection import ObjectDetection
import os



    
    
    
class ObjectDetection:


    def __init__(self, path, out):
        self.model = self.load_model()
        self.classes = self.model.names
        self.out = out
        self.device = 'cpu'
        self.path=path
        
    def get_video(self):
        return cv2.VideoCapture(self.path)

    def get_image(self):
        images = []
        names=[]
        for i in glob.glob(self.path+"/*.png"):
            image=cv2.imread(i)
            images.append(image)
            names.append(os.path.basename(i))
        images = np.asarray(images)
        return images,names
    
    def load_model(self):

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    


    def score_frame(self, frame):

        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame,size=640)
        labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        return labels, cord

    def class_to_label(self, x):

        return self.classes[int(x)]

    def plot_boxes(self, results, frame):

        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    def __call__(self,typeOfinput):
        
        if typeOfinput=='video':
            player = self.get_video()
            assert player.isOpened()
            x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
            y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
            four_cc = cv2.VideoWriter_fourcc(*"MPEG")
            out = cv2.VideoWriter(self.out+"/videoaftercv/result.mp4", four_cc, 20, (x_shape, y_shape))
            while True:
                start_time = time()
                ret, frame = player.read()
                if ret:
                    results = self.score_frame(frame)
                    frame = self.plot_boxes(results, frame)
                    
                    
                    end_time = time()
                    print('Processing real-time frames with Object Detection')
                    fps = 1/np.round(end_time - start_time, 3)
                    out.write(frame)
                else:
                    break
            out.release()
        else:
            images,names=self.get_image()
            i=0
            execution_path = os.getcwd()
            for image in images:
                results = self.score_frame(image)
#                 detector = ObjectDetection(path=os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"),out='')
#                 returned_image, detections = detector.detectObjectsFromImage(input_image=image, input_type=array, output_type=array)

#                 for eachObject in detections:
#                     print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

            
            
            
                image = self.plot_boxes(results, image)
                cv2.imwrite(self.out+"/"+names[i],image)
                i=i+1
                
                
        
            

