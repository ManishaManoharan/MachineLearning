import cv2
import os
import numpy as np

path = "dataset"

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
names = {}

label_id = 0

for person in os.listdir(path):

    names[label_id] = person

    person_path = os.path.join(path, person)

    for image in os.listdir(person_path):

        img_path = os.path.join(person_path, image)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        faces.append(img)
        labels.append(label_id)

    label_id += 1

recognizer.train(faces, np.array(labels))

recognizer.save("face_model.yml")

print("Training Completed")