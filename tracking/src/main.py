from turtle import back
import cv2
import mediapipe as mp
from time import time
from math import hypot

import render

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

nose_image = cv2.imread("pig-nose.png")
background_image = cv2.imread("pastel-bkg.jpg")
mask_image = cv2.imread("mask.png")

with mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    time1 = 0
    # 0 is none, 1 is landmarks, 2 is mesh
    render_mode = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'
            continue

        # Flip the image for "selfie" mode
        image = cv2.flip(image, 1)
        
        # To improve performance, optionally mark the image as not writeable to pass by reference
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # FPS tracker
        time2 = time()
        if time2-time1 > 0:
            fps = 1.0 / (time2-time1)
            cv2.putText(image, 'FPS: {}'.format(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        time1 = time2

        # Color correction
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        frame = background_image.copy()
        
        # Rendering
        if results.multi_face_landmarks:
            if render_mode == 0:
                render.render_mask(results, image, mask_image)
            elif render_mode == 1:
                render.render_landmarks(results, image)
            elif render_mode == 2:
                render.render_mesh(results, image, mp_drawing, mp_face_mesh, mp_drawing_styles)  

        # Show specific landmarks
        # landmark_list = [64, 294, 195]
        # if results.multi_face_landmarks:
        #     for i in landmark_list:
        #         landmark = results.multi_face_landmarks[0].landmark[i]
        #         x = landmark.x
        #         y = landmark.y

        #         shape = image.shape
        #         top_nose = (int(x*shape[1]), int(y*shape[0]))

        #         cv2.circle(image, top_nose, radius=1, color=(255, 0, 100), thickness=2)

        # Toggle render modes with spacebar
        if cv2.waitKey(5) == 32:
            if render_mode == 0:
                render_mode = 1
            elif render_mode == 1:
                render_mode = 2
            elif render_mode == 2:
                render_mode = 0
            else:
                render_mode = 0

        # Flip the image horizontally for a selfie-view display
        cv2.imshow('asldkfj', image)
        # cv2.imshow("final", frame)
        # Use ESC to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()