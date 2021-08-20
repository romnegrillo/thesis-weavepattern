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

class ThesisTextileRecognition:

    def __init__(self,webcam=True,rpicam=False):

        self.webcam=webcam
        self.rpicam=rpicam
        self.view_num=1

        try:
            if self.webcam:
                self.capture=cv2.VideoCapture(0)
            elif self.rpicam:
                pass
        except Exception as exp:
            print(str(exp))

    def get_frames(self):
        if self.webcam:
            t,self.frame=self.capture.read()
        else:
            pass

        bgr=self.frame
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        t,tresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        if self.view_num == 1:
            #print("rgb")
            return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        elif self.view_num == 2:
            #print("gray")
            return gray
        elif self.view_num == 3:
            #print("threshold")
            return tresh

        return cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)

    def close_cam(self):
        if self.webcam:
            self.capture.release()
        elif self.rpicam:
            return

    def change_view(self,view_num):
        self.view_num=view_num
