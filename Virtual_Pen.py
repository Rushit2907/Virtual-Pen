import cv2
import numpy as np
import time

load_from_disk =True

if load_from_disk:
    penval = np.load('penval.npy')

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

kernel = np.ones((5,5),np.uint8)

canvas = None
x1,y1 = 0,0

noiseth = 800
wiper_thresh = 40000
clear = False

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    if load_from_disk:
        lower_range = penval[0]
        upper_range = penval[1]
    else :
        lower_range =np.array([26,80,147])
        upper_range=np.array([81,255,255])

    mask = cv2.inRange(hsv,lower_range,upper_range)

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask,kernel,iterations=2)

    contours,hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours,key=cv2.contourArea)) >noiseth:
        c=max(contours,key=cv2.contourArea)
        x2,y2,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        if x1 == 0 and y1 == 0:
            x1,y1=x2,y2
        else:
            canvas = cv2.line(canvas,(x1,y1),(x2,y2),[255,0,0],5)
        x1, y1 = x2, y2
        if area>wiper_thresh:
            cv2.putText(canvas,'cleaning Canvas',(100,200),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5,cv2.LINE_AA)
            clear = True
    else:
        x1, y1 = 0,0

    _,mask = cv2.threshold(cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY),20,255,cv2.THRESH_BINARY)

    foreground = cv2.bitwise_and(canvas,canvas,mask=mask)
    background = cv2.bitwise_and(frame,frame,mask=cv2.bitwise_not(mask))
    frame = cv2.add(foreground,background)

    cv2.imshow('image',frame)

    k = cv2.waitKey(5) & 0xFF
    if k==27:
        break
    if clear == True:
        time.sleep(1)
        canvas = None
        clear = False

cap.release()
cv2.destroyAllWindows()