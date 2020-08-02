import face_recognition as fr
import numpy as np
import cv2

ref_image = fr.load_image_file('known_faces_dir/Harman/ref_image.jpg')  # must save a ref_image.jpg into the working directory
encoding = fr.face_encodings(ref_image)[0]

known_face_encodings = [encoding]
known_face_names = ['Harman']  # Must put desired name here

video = cv2.VideoCapture(0)  # take video from webcam

while True:
    ret, frame = video.read()
    color_corrected = frame[:, :, ::-1]
    face_locations = fr.face_locations(color_corrected)  # using hog by default because faster
    face_encodings = fr.face_encodings(color_corrected, face_locations)

    for face_location, face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = 'Unrecognized'
        disparity = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(disparity)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        top = face_location[0]
        right = face_location[1]
        bottom = face_location[2]
        left = face_location[3]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('Webcame_facerecognition', frame)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

video.realease()
cv2.destroyAllWindows()
