import cv2
import face_recognition as fr
from PIL import Image, ImageDraw
import numpy as np

vc = cv2.VideoCapture(0)

while True:
    ret, frame = vc.read()

    rgb_frame = frame[:,:,::-1]

    face_loc = fr.face_locations(rgb_frame)
    face_encoding = fr.face_encodings(rgb_frame, face_loc)


    for(t, r, b, l), face_encoding in zip(face_loc, face_encoding):
        cv2.rectangle(frame, (l,t), (r,b), (0,255,0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
