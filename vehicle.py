import cv2 #importing libraries
import numpy as np


#Web camera
cap = cv2.VideoCapture('video.mp4') #path of video

min_width_react=80 #minimum width of a rectangle
min_height_react=80 #minimum height of a rectangle

count_line_position = 550
#Initialize Substructor
algo = cv2.bgsegm.createBackgroundSubtractorMOG() #used to subtract the background from the image

#to detct the centre point of the vehicle 
def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx= x+x1
    cy= y+y1
    return cx,cy 

detect = []
offset=6 #Allowing error between pixel    
counter=0


while True:
    ret,frame1 = cap.read() #to read the video
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY) #to convert to grey image
    blur = cv2.GaussianBlur(grey,(3,3),5) 
    #applying on each frame
    img_sub = algo.apply(blur)
    dilate = cv2.dilate(img_sub,np.ones((5,5))) #determines the shape of a pixel neighborhood 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) #return structuring element of the specified size and shape
    dilatada = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    counterShape,h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #contours in a binary image

    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),3) #to draw the line in the video 
   
   #for bounding box
    for (i,c) in enumerate(counterShape):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_react) and (h>= min_height_react)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1,"Vehicle"+str(counter),(x, y-20), cv2.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2)



        center= center_handle(x,y,w,h)
        detect.append(center) #to detct the centre point of the vehicle 
        cv2.circle(frame1,center,4, (0,0,255),-1) 

        #to count no.of vehicles after passing through the line
        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1
                cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
                detect.remove((x,y))

                print("Vehicle Counter:"+str(counter))




    cv2.putText(frame1,"VEHICLE COUNTER :"+str(counter),(450,70),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),5)

   
    #cv2.imshow('Detector',dilatada)
    cv2.imshow('Video Original',frame1) #to show the video

    if cv2.waitKey(1) == 13: #to close the window
        break

cv2.destroyAllWindows()
cap.release()
