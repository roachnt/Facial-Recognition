# OpenCV Library
import cv2
# System Library
import sys
# Face Recognition Library
import face_recognition
# Python Image Library
from PIL import Image

cv2.useOptimized()
# Get haar cascade file
cascPath = "haarcascades/haarcascade_frontalface_default.xml"

# Get my picture
picture_of_me = face_recognition.load_image_file("img/picture_of_me.jpg")

# Encode face
my_face_encoding = face_recognition.face_encodings(picture_of_me)[0]

# Create the haar cascadeOB
faceCascade = cv2.CascadeClassifier(cascPath)

# Get webcam video, width, height
video_capture = cv2.VideoCapture(0)
frameWidth = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    # Read frame of video
    ret, frame = video_capture.read()
    
    # Gray color of image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get faces in image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi = frame[y:y+h, x:x+w]
        if face_recognition.face_encodings(roi):
            unknown_face_encoding = face_recognition.face_encodings(roi)[0]
            results_me = face_recognition.compare_faces([my_face_encoding], unknown_face_encoding)
            if results_me[0]:
                cv2.putText(frame, 'Nick Roach', (x - 1, y - 1), cv2.FONT_HERSHEY_TRIPLEX,2,(0, 0, 255))
        # Display the resulting frame
    cv2.imshow('Facial Recognition Application', frame)

                
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
