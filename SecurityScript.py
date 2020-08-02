import face_recognition as fr
import numpy as np
import pandas as pd
import cv2
import datetime as dt
import os

# Store all known names and known faces
known_faces_dir = 'known_faces'
known_faces = []
known_names = []

for name in os.listdir(known_faces_dir):  # iterate through each file in the known_faces
    for filename in os.listdir(f'{known_faces_dir}/{name}'):  # iterate through files in a known name
        image = fr.load_image_file(f'{known_faces_dir}/{name}/{filename}')
        encoding = fr.face_encodings(image)[0]  # list of lists, want first entry
        known_faces.append(encoding)
        known_names.append(name)

# keep_indices = pd.DataFrame(data=known_names).T
# keep_indices.columns = known_names
d2 = {'Name': [0], 'Time of Entry': [0], 'Left at': [0]}
occupants = pd.DataFrame(data=d2)

video = cv2.VideoCapture(0)  # take video from webcam

date = dt.today.strftime('%Y-%m-%d')

while True:
    ret, frame = video.read()
    color_corrected = frame[:, :, ::-1]
    face_locations = fr.face_locations(color_corrected, model='cnn')  # using hog by default because faster
    face_encodings = fr.face_encodings(color_corrected, face_locations)
    time = dt.now().strftime("%H:%M:%S")
    for face_location, face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_faces, face_encoding)  # list of boolean variables
        name = 'Unrecognized'
        disparity = fr.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(disparity)
        if matches[best_match_index]:
            name = known_faces[best_match_index]
            entry_time = time
            occupants.append(pd.DataFrame(data={'Name': [name], 'Time of Entry': [entry_time], 'Left at': [np.nan]}))
        else:
            index = 0
            occupants.loc[index, 'Left at'] = entry_time
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
