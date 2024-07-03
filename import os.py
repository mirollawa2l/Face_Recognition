import os
import cv2 as cv
import numpy as np

people = ['Elon Mask', 'Bill Gates', 'Sundar Pichai', 'Jeff Bezos', 'Mark Zuckerberg', 'Dara Khosrowshahi']
DIR = r'C:\Codes\Linear face recognition project T3\Resource'
haar_cascade = cv.CascadeClassifier('haar_face.xml')

features = []
labels = []

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # Convert to grayscale
            if img_array is None:
                print(f'Error reading image: {img_path}')
                continue 

            faces_rect = haar_cascade.detectMultiScale(img_array, scaleFactor=1.3, minNeighbors=4)

            for (x, y, w, h) in faces_rect:
                faces_roi = img_array[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training done ')

features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv.face.LBPHFaceRecognizer_create()

face_recognizer.train(features, labels)

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)