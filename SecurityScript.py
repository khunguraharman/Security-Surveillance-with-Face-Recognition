import face_recognition as fr
import numpy as np
import pandas as pd
import cv2
import datetime as dt
import os

if not os.path.exists('Security Logs'):
    os.mkdir('Security Logs')

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
columns = ['Name', 'Time of Entry', 'Left at']
indices = range(0, 1, 1)
security_log = pd.DataFrame(index=indices, columns=columns)
test = pd.DataFrame(data={'Name': ['Harman'], 'Time of Entry': [0], 'Left at': [0]})
security_log = security_log.append(test)
security_log = security_log.reset_index(drop=True)

instance = dt.datetime.now()
today = instance.date().strftime("%Y-%m-%d")
prev_frame_occupants = set()

video = cv2.VideoCapture(0)  # take video from webcam
record = True
if not os.path.exists(f'Security Logs/{today}'):
    os.mkdir(f'Security Logs/{today}')
print ("After cv2.VideoCapture(0): cap.grab() --> " + str(cap.grab()) + "\n")
while record:
    ret, frame = video.read()
    color_corrected = frame[:, :, ::-1]
    face_locations = fr.face_locations(color_corrected, model='cnn')  # using hog by default because faster
    face_encodings = fr.face_encodings(color_corrected, face_locations)
    time = instance.time().strftime("%H:%M:%S")
    current_occupants = set()  # don't know who is currently in the room yet
    for face_location, face_encoding in zip(face_locations, face_encodings):  # get current occupants
        matches = fr.compare_faces(known_faces, face_encoding)  # list of boolean variables
        disparity = fr.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(disparity)
        name = 'Unrecognized'
        if matches[best_match_index]:  # recognized a face, room initially empty
            name = known_names[best_match_index]
            current_occupants.add(name)
        top = face_location[0]
        right = face_location[1]
        bottom = face_location[2]
        left = face_location[3]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    entered = current_occupants.difference(prev_frame_occupants)  # who entered the room
    for name in entered:
        security_log = security_log.append(pd.DataFrame(data={'Name': [name], 'Time of Entry': [time], 'Left at': [0]}))
        security_log = security_log.reset_index(drop=True)
    left = prev_frame_occupants.difference(current_occupants)  # who left the room
    for name in left:
        index = security_log.loc[(security_log['Name'] == name) & (security_log['Left at'] == 0)].index.to_list()[0]
        security_log.loc[index, ['Left at']] = time
    prev_frame_occupants = current_occupants

    cv2.imshow('Surveillance', frame)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        record = False
        break

video.realease()
cv2.destroyAllWindows()
security_log.to_csv(f'Security Logs/{today}.csv')

