import cv2
import mediapipe as mp
from pynput.mouse import Controller
import numpy as np

#initialises mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence= 0.5)
mp_drawing = mp.solutions.drawing_utils

#Initialise pynput mouse controller
mouse = Controller()

#Set up Camera
cap = cv2.VideoCapture(0)

#Screen Dimention
screen_width, screen_height = 1920,1080

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            #Extract the coordinates of the eyes (index 33 & 263 are example landmarks)
            left_eye = landmarks.landmark[33]
            right_eye = landmarks.landmark[263]

            #convert normalised coordinates to pixel coordinates
            h,w,_ = frame.shape
            left_eye_x, left_eye_y = int(left_eye.x*w), int(left_eye.y*h)
            right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

            #Average of both eye positions
            eye_x = (left_eye_x + right_eye_x) //2
            eye_y = (left_eye_y + right_eye_y) //2

            #Map eye position to screen coordinates
            mouse_x = int(screen_width * (eye_x/w))
            mouse_y = int(screen_height * (eye_y/h))

            #Move the mouse
            mouse.position = (mouse_x, mouse_y)

            #Draw landmarks for visualisation
            #mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACE_CONNECTIONS)

    #Show the frame
    cv2.imshow('Eye Tracker', frame)

    #Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


