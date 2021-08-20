import cv2

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
