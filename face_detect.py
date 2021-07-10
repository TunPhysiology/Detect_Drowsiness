import numpy as np
import cv2


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0,  cv2.CAP_DSHOW)
#cap = VideoStream(src=0).start()
#cap.set(3,640) # set Width
#cap.set(4,480) # set Height


############################################
def detectface(): 
    while(True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
        if ret == True:
        	for (x, y, w, h) in faces:
        		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)


        cv2.imshow('frame', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
        	break
detectface()
###############################################




cap.release()
cv2.destroyAllWindows()