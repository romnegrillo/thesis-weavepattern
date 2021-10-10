from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.uic import loadUi
import sys
from ThesisTextileRecognition import ThesisTextileRecognition 
import time


in_rpi = False
try:
    import RPi.GPIO
    in_rpi = True
except Exception as exp:
    print(str(exp))


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(MainWindow,self).__init__()
        loadUi("mainwindow.ui",self)

        self.capture_button.clicked.connect(self.capture_button_clicked)
        self.reset_button.clicked.connect(self.reset_button_clicked)

        self.class_textbox.setText("-")
        self.confidence_textbox.setText("-")

        if in_rpi:
            self.imgObject=ThesisTextileRecognition("mobile_net_v2_test/cats_and_dogs_mobilenetv2.h5",
                                                    use_webcam = False,
                                                    use_rpicam = True)
            self.showFullScreen()
        else:
            self.imgObject=ThesisTextileRecognition("mobile_net_v2_test/cats_and_dogs_mobilenetv2.h5",
                                                    use_webcam=True,
                                                    use_rpicam = False)
                                                    
            self.showFullScreen()

        self.is_captured = False

        self.timer=QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(5)
 
    def capture_button_clicked(self):
        print("Capture button clicked.")

        if not self.is_captured:
            self.is_captured = True
            self.timer.stop()

    def reset_button_clicked(self):
        print("Reset button clicked.")
        
        if self.is_captured:
            self.is_captured = False
            self.timer.start(5)


    def update_frames(self):
        self.image=self.imgObject.get_frames()

        # If there is only 2 items in shape, it means the
        # image is one channel.
        if(len(self.image.shape)==2):
            imageFormat=QtGui.QImage.Format_Indexed8
        # Else, it may be 3 or 4
        else:
            # Get third item which is the number of channels.
            num_channels=self.image.shape[2]
            if num_channels==1:
                #print("Debug1")
                imageFormat=QtGui.QImage.Format_Indexed8
            elif num_channels==3:
                #print("Debug2")
                imageFormat=QtGui.QImage.Format_RGB888
            elif num_channels==4:
                #print("Debug3")
                imageFormat=QtGui.QImage.Format_RGBA8888

        outImage=QtGui.QImage(self.image,self.image.shape[1],self.image.shape[0],self.image.strides[0],imageFormat)

        self.image_label.setPixmap(QtGui.QPixmap.fromImage(outImage))
        self.image_label.setScaledContents(True)

if __name__=="__main__":

    app=QtWidgets.QApplication(sys.argv)
    w=MainWindow()
    w.show()
    app.exec_()
