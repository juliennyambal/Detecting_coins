import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#img = cv2.imread('testing/t4.jpg',0)

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #img = cv2.medianBlur(frame,5)
    # Our operations on the frame come here
    #cimg = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)

    img = frame[:,:,0]
    
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=40,maxRadius=60)

    
    
    if circles is not None:
#        print circles
        for i in circles[0,:]:
            # draw the outer circle
            frame = frame.copy()
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
            cv2.putText(frame,str(i[2]),(i[0],i[1]),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
    else:
        print 'out'
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)
       
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=40,maxRadius=60)


for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    cv2.putText(cimg,str(i[2]),(i[0],i[1]),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)



test = np.array([circles[0][0],
                circles[0][1],
                circles[0][2],
                circles[0][3],
                circles[0][4]]),
                circles[0][5],
                circles[0][6],
                circles[0][7]])
#                circles[0][8]]
#                circles[0][9]])
te = np.savetxt('t4.csv',test,delimiter=',',fmt='%.5f')

cv2.namedWindow('detected circles',cv2.WINDOW_NORMAL)
cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()