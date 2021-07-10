import cv2
import dlib
from scipy.spatial import distance
import numpy as np



faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0,  cv2.CAP_DSHOW)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def detectFace():
    faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
    if ret == True:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)


def tinhtilemat(eye):
    A = distance.euclidean(eye[1],eye[5])
    B = distance.euclidean(eye[2],eye[4])
    C = distance.euclidean(eye[0],eye[3])
    ear = (A+B)/(C*2.0)
    return ear   #eye aspect ratio





while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    for face in faces:

        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []


        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x,y))
            next_point = n + 1
            if n == 41:
                next_point=36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y),(x2,y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x,y))
            next_point = n + 1
            if n == 47:
                next_point=42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            cv2.line(frame, (x, y),(x2,y2), (0, 255, 0), 1)

        left_ear = tinhtilemat(leftEye)
        right_ear = tinhtilemat(rightEye)
        EAR = (left_ear + right_ear)/2
        EAR = round(EAR,2)
        print(EAR)

        if EAR < 0.2:
            cv2.putText(frame,"WARINING",(10,30),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),3)


    detectFace()
    cv2.imshow("tuan", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):   #bam q trên bàn phím để thoát
        break
cap.release()
cv2.destroyAllWindows()