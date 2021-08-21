# CpE Thesis - Philippines Indigenous Textile Pattern Recognition using Convolutional Neural Network

In progress...

### Language and Libraries used in Development in Ubuntu
* Python 3
* Tensorflow
* OpenCV
* Numpy
* Qt5

Python modules are packed in requirements.txt.

In Raspberry Pi 4, you can't use the module versions created 
from Ubuntu's requirements.txt, you have to install the commands
below manually again:
* sudo apt update
* sudo apt upgrade
* reboot
* pip3 install https://github.com/bitsy-ai/tensorflow-arm-bin/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl
* pip3 install tensorflow_hub
* pip3 install opencv-python
* pip3 install matplotlib
* pip3 install numpy
* pip3 install numpy --upgrade
* pip3 install seaborn
* sudo apt-get install python3-pyqt5
* sudo apt-get install libatlas-base-dev
* sudo apt-get install libjasper-dev
* sudo apt-get install libqtgui4
* sudo apt install libqt4-test
* reboot
* sudo modprobe bcm2835-v4l2  (For enabling USB camera.)
* RPI camera can be enable in sudo raspi-config.


### To Buy:
* https://shopee.ph/Raspberry-Pi-Camera-Module-V2-8-Megapixels-i.18252381.252067660?position=2 - 1649
(Trambia)
* https://shopee.ph/product/43891586/5656778897?smtt=0.92093202-1629530374.9
(Pinilian)
* https://shopee.ph/product/273736909/5256813649?smtt=0.92093202-1629531631.9
