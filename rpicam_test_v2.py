import io
import time
import picamera
import cv2
import numpy as np

# Create the in-memory stream
stream = io.BytesIO()
camera = picamera.PiCamera()
camera.start_preview()
time.sleep(2)


camera.capture(stream, format='jpeg')
# Construct a numpy array from the stream
data = np.fromstring(stream.getvalue(), dtype=np.uint8)
# "Decode" the image from the array, preserving colour
image = cv2.imdecode(data, 1)
# OpenCV returns an array with data in BGR order. If you want RGB instead
# use the following...
#image = image[:, :, ::-1]
cv2.imshow("test", image)
cv2.waitKey(0)
camera.close()
