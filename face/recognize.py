import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

names = os.listdir("dataset")

cam = cv2.VideoCapture(0)

while True:

    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face = gray[y:y+h,x:x+w]

        label, confidence = recognizer.predict(face)

        if confidence < 70:
            name = names[label]
        else:
            name = "Face Not Registered"

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        cv2.putText(frame,name,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

    cv2.imshow("Face Recognition",frame)

    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()