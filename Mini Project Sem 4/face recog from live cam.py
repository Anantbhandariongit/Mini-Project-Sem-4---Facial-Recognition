import cv2
import face_recognition as fr



vc = cv2.VideoCapture(0)
anant_img = fr.load_image_file("Anant Bhandari.jpg")
anant_face_encoding = fr.face_encodings(anant_img)[0]

mom_img = fr.load_image_file("mataji.jpg")
mom_face_encoding = fr.face_encodings(mom_img)[0]

dad_img = fr.load_image_file("pitaji.jpg")
dad_face_encoding = fr.face_encodings(dad_img)[0]


known_face_encodings = [
        anant_face_encoding,
        mom_face_encoding,
        dad_face_encoding  
     ]

known_face_names = [
    "123", "Upma", "Sunil"
]


while True:
    ret, frame = vc.read()

    rgb_frame = frame[:,:,::-1]

    face_loc = fr.face_locations(rgb_frame)
    face_encoding = fr.face_encodings(rgb_frame, face_loc)


    for(t, r, b, l), face_encoding in zip(face_loc, face_encoding):
        matches = fr.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        if True in matches:
            first_match = matches.index(True)
            name = known_face_names[first_match]

        cv2.rectangle(frame, (l, b -  35), (r,b), (0,0,255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (l + 6, b - 6), font, 1.0, (255,255,255,255), 1)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()










































