import cv2 as cv
import numpy as np 

Cascade = cv.CascadeClassifier('haar_face.xml')
faces = ['Elon Musk', 'Bill Gates', 'Sundar Pichai', 'Jeff Bezos', 'Mark Zuckerberg', 'Dara Khosrowshahi']

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

image = cv.imread("C:/Codes/Linear face recognition project T3/Resource.jpg")  # Use forward slashes for file paths

gray_v = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('faces', gray_v)

rect = Cascade.detectMultiScale(gray_v, 1.3, 1)

for (x, y, width, height) in rect:
    region_of_interest = gray_v[y:y+height, x:x+width]

    prediction, confidence_level = face_recognizer.predict(region_of_interest)

    print(f'Prediction: {faces[prediction]}, Confidence: {confidence_level}')

    cv.putText(image, f'{faces[prediction]}', (x, y-10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(image, (x, y), (x+width, y+height), (100, 0, 255), thickness=2)

cv.imshow('Detected Face', image)

cv.waitKey(0)