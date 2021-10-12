from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import time

# Initialize objects.
# Camera object and PiRGBArray for converting the
# image returned by the picamera into array of pixel values.
camera = PiCamera(framerate=10)
rawCapture = PiRGBArray(camera)

# Warmup
time.sleep(2)

# Starting capturing, make sure the format is in bgr
# because OpenCV by default, the displayed image is in bgr.

# Inifite loop
while 1:
    camera.capture(rawCapture, format="bgr")

    # Get the array of pixels from the captured image.
    image=rawCapture.array
    rawCapture.truncate(0)
 
    #print(image)

    # Flip the display because PiCamera was placed up side down.
    # Display the image on screen and wait for a keypress.
    #image=cv2.flip(image,0)
    cv2.imshow("Image", image)
     
    # Break if q is pressed.
    if cv2.waitKey(1) == ord('q'):
        break
    
camera.close()
cv2.destroyAllWindows()
