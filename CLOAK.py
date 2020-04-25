#importing libraries needed
import cv2
import numpy as np
import time
#capturing the video
video_capture = cv2.VideoCapture(0)


#let the camera warm up
time.sleep(3)
background = 0

#capturing the background
for k in range(45):
    ret,background = video_capture.read()

#The Main loop
while(video_capture.isOpened()):
    ret, frame = video_capture.read()
    if not ret:
        break
    
    
    # convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Create masks with ranges of blue color
    lower_blue = np.array([100, 80, 2])
    upper_blue = np.array([126, 255, 255])
    mask_all = cv2.inRange(hsv,lower_blue,upper_blue)

    #Do morphological operations
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    mask_all = cv2.morphologyEx(mask_all, cv2.MORPH_DILATE, np.ones((3,3),np.uint8))
 
 
    #Hide the blue part a
    mask2 = cv2.bitwise_not(mask_all)
 
    streamA = cv2.bitwise_and(frame,frame,mask=mask2)

    #Copy the masked area's original part
    streamB = cv2.bitwise_and(background, background, mask = mask_all)
 
    #Generating the final output
    output = cv2.addWeighted(streamA,1,streamB,1,0)

    cv2.imshow("Thankyou_Dumbledore",output)
    if cv2.waitKey(25) == 13:
        break

video_capture.release()
cv2.destroyAllWindows()