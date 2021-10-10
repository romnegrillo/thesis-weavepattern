import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras

labels = ["checkered pattern", "dotted pattern", "floral pattern",  "solid pattern", "stripped pattern", "zig zag"]

new_model = tf.keras.models.load_model("cloth_pattern_mobilenetv2.h5")
new_model.summary()

vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    
    frame_filtered = cv2.resize(frame, (160,160))
    #frame_filtered = frame_filtered/255.0
   

    predictions_list = new_model.predict(np.array([frame_filtered]))
    prediction = np.argmax(predictions_list[0])
    proba = np.round(float(predictions_list[0][prediction]) * 100, 2)   

    print(predictions_list)
    label = ""

    if proba >= 90:
        label = labels[prediction]
        print("{} %".format(proba))
    else:
        label = "Unknown"

    cv2.putText(frame, label, (10,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()