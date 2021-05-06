import cv2
import numpy as np

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

cv2.namedWindow("TrackBars")

cv2.createTrackbar("L -H","TrackBars",0,179,nothing)
cv2.createTrackbar("L -S","TrackBars",0,255,nothing)
cv2.createTrackbar("L -V","TrackBars",0,255,nothing)
cv2.createTrackbar("U -H","TrackBars",179,179,nothing)
cv2.createTrackbar("U -S","TrackBars",255,255,nothing)
cv2.createTrackbar("U -V","TrackBars",255,255,nothing)

while True:
    ret ,frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    l_h = cv2.getTrackbarPos("L -H","TrackBars")
    l_s = cv2.getTrackbarPos("L -S","TrackBars")
    l_v = cv2.getTrackbarPos("L -V","TrackBars")
    u_h = cv2.getTrackbarPos("U -H","TrackBars")
    u_s = cv2.getTrackbarPos("U -S","TrackBars")
    u_v = cv2.getTrackbarPos("U -V","TrackBars")

    lower_range = np.array([l_h,l_s,l_v])
    upper_range = np.array([u_h,u_s,u_v])

    mask = cv2.inRange(hsv,lower_range,upper_range)
    res = cv2.bitwise_and(frame,frame,mask=mask)

    mask_3 = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    stacked = np.hstack((mask_3,frame,res))

    cv2.imshow('Trackbars', cv2.resize(stacked, None, fx=0.4, fy=0.4))

    key = cv2.waitKey(1)
    if key ==27:
        break

    if key==ord('k'):
        thearray = [[l_h,l_s,l_v],[u_h,u_s,u_v]]
        print(thearray)

        np.save('penval',thearray)
        break

cap.release()
cv2.destroyAllWindows()