from cv2 import cv2   #modules
import numpy as np
import math

capture=cv2.VideoCapture(0)    #configs camera

while capture.isOpened():     #runs prog untill we switch the cam off
    ret,frame = capture.read()

    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)  #forms rectangle where signs are read.
    crop_img = frame[100:300,100:300]

    blur= cv2.GaussianBlur(crop_img,(3,3),0) #apply noise reduc filter

    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)

    mask2=cv2.inRange(hsv,np.array([2,0,0]),np.array([20,255,255]))  #creates box that displys hand detection in binary

    kernel = np.ones((5,5))  #kernel for ilussion and dillusion

    dilation=cv2.dilate(mask2,kernel,iterations=1)   #filters noise
    erosion =cv2.erode(dilation,kernel,iterations=1)

    filtered=cv2.GaussianBlur(erosion,(3,3),0)   #apply Gaus filter and threshold
    ret,thresh=cv2.threshold(filtered,127,255,0)

    cv2.imshow("THRESHOLDED",thresh) #show threshold img

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #finding contours

    try:
        contour=max(contours,key=lambda x: cv2.contourArea(x))#find contour with max ar

        x,y,w,h=cv2.boundingRect(contour)   #make boundry rectangle arnd the contour
        cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)

        hull=cv2.convexHull(contour)    #convex hull find

        draw=np.zeros(crop_img,np.uint8)  #draw the contour
        cv2.drawContours(draw,[contour],-1,(0,255,0),0)
        cv2.drawContours(draw[hull], -1,(0,0,255),0)

        hull=cv2.convexHull(contour,returnPoints=False)   #find convexity defects
        defects=cv2.convexityDefects(contour,hull)

        count_defects =0  #using cosine rule to find the change in angle if hand is moved,from start to end point.

        for i in range(defects.shape[0]):
            s,e,f,d=defects[i,0]
            start =tuple(contour[s][0])
            end =tuple(contour[e][0])
            far =tuple(contour[f][0])

            a= math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
            b= math.sqrt((end[0]-start[0])**2+(far[1]-start[1])**2)
            c= math.sqrt((end[0]-far[0])**2+(end[1]-far[1])**2)
            angle=(math.acos((b**2+c**2-a**2)/(2*b*c))*180)/3.14


            if angle<=90:   #if angle is <= 90 draw cirlc at far point.
                count_defects +=1
                cv2.circle(crop_img,far,1,[0,0,255],-1)
            cv2.line(crop_img,start,end,[0,255,0],2)
        
        #print no. of fingerss
        if count_defects==0:
            cv2.putText(frame,"ONE",(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        elif count_defects==1:
            cv2.putText(frame,"TWO",(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        elif count_defects==2:
            cv2.putText(frame,"THREE",(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        elif count_defects==3:
            cv2.putText(frame,"FOUR",(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        elif count_defects==4:
            cv2.putText(frame,"FIVE",(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        else:
            pass
    except:
        pass

    cv2.imshow("Gesture",frame)  #to show required image
    all_img=np.hstack((draw,crop_img))
    cv2.imshow('Contours',all_img)

    if cv2.waitKey(1)== ord("q"):  #to exit the screen,press"q"
        break

capture.release()
cv2.destroyAllWindows()