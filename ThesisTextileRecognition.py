import cv2

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray

class ThesisTextileRecognition:

    def __init__(self, trained_model_path, use_webcam=True,use_rpicam=False):

        self.use_webcam=use_webcam
        self.use_rpicam=use_rpicam
        self.view_num=1
        self.trained_model_path = trained_model_path
        self.model = None
        self.labels = ["checkered pattern", "dotted pattern", "floral pattern",  "solid pattern", "stripped pattern", "zig zag"]

        
        self.load_model(self.trained_model_path)

        try:
            if self.use_webcam:
                self.capture=cv2.VideoCapture(0)
            elif self.use_rpicam:
                self.camera = PiCamera(framerate=10)
                self.rawCapture = PiRGBArray(self.camera)
        except Exception as exp:
            print(str(exp))

    def get_frames(self):
        if self.use_webcam:
            t,self.frame=self.capture.read()
        elif self.use_rpicam:
            self.camera.capture(self.rawCapture, format="bgr", use_video_port=True)
            self.frame=self.rawCapture.array

        bgr=self.frame.copy()

        if self.use_rpicam:
            self.rawCapture.truncate(0)
            
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        t,tresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        self.get_prediction(bgr)
        
        if self.view_num == 1:
            #print("rgb")
            return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        elif self.view_num == 2:
            #print("gray")
            return gray
        elif self.view_num == 3:
            #print("threshold")
            return tresh

    def close_cam(self):
        if self.use_webcam:
            self.capture.release()
        elif self.use_rpicam:
            return
        
    def change_view(self,view_num):
        self.view_num=view_num

    def load_model(self, trained_model_path):
        self.model = tf.keras.models.load_model(trained_model_path)
        self.model.summary()

    def get_prediction(self, input_image):

        #print("DEBUG")
        
        frame_filtered = cv2.resize(input_image, (160,160))
        #frame_filtered = frame_filtered/255.0

        predictions_list = self.model.predict(np.array([frame_filtered]))
        prediction = np.argmax(predictions_list[0])
        proba = np.round(float(predictions_list[0][prediction]) * 100, 2)   

        #print(predictions_list)
        label = ""

        if proba >= 90:
            label = self.labels[prediction]
            #print("{} %".format(proba))
        else:
            label = "Unknown"

        #print(label)

        
