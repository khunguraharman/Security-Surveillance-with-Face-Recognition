import face_recognition
import os
import cv2

known_faces_dir = 'known_faces'
unknown_faces_dir = 'unknown_faces'
tolerance = 0.4
frame = 2
font = 2
model = 'cnn'

print('loading known faces')
known_faces = []
known_names = []

for name in os.listdir(known_faces_dir):  # iterate through each file/filename in the directory
    for filename in os.listdir(f'{known_faces_dir}/{name}'):  # iterate through files in a known name
        image = face_recognition.load_image_file(f'{known_faces_dir}/{name}/{filename}')
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

print('processing unknown faces')

for filename in os.listdir(unknown_faces_dir):
    print(filename)
    image = face_recognition.load_image_file(f'{unknown_faces_dir}/{filename}')
    locations = face_recognition.face_locations(image, model=model)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, tolerance)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f'MAtch Found: {match}')
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, frame)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putTest(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), font)

    cv2.imshow(filename, image)
    cv2.waitKey(10000)
    cv2.destroyWindow(filename)
