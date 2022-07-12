from math import hypot, atan, pi
import cv2

def render_mesh(results, image, mp_drawing, mp_face_mesh, mp_drawing_styles):
    # Draw the face mesh annotations on the image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

def render_landmarks(results, image):
    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            for landmark in face.landmark:
                x = landmark.x
                y = landmark.y

                shape = image.shape
                relative_x = int(x * shape[1])
                relative_y = int(y * shape[0])
                cv2.circle(image, (relative_x, relative_y), radius=1, color=(225, 0, 100), thickness=2)

def render_pig_nose(results, image, nose_image, background_image, size):
    for face in results.multi_face_landmarks:
        shape = image.shape
        
        top_nose = (int(face.landmark[195].x*shape[1]), int(face.landmark[195].y*shape[0]))
        left_nose = (int(face.landmark[64].x*shape[1]), int(face.landmark[64].y*shape[0]))
        right_nose = (int(face.landmark[294].x*shape[1]), int(face.landmark[294].y*shape[0]))
        center_nose = (int((left_nose[0]+right_nose[0])/2), int(face.landmark[4].y*shape[0]))

        # nose_width = int(hypot(abs(left_nose[0]-right_nose[0]), abs(left_nose[1]-right_nose[1])) * 1.3)
        nose_width = size

        top_left = (int(center_nose[0] - nose_width / 2), int(center_nose[1] - nose_width / 2))
        # bottom_right = (int(center_nose[0] + nose_width / 2), int(center_nose[1] + nose_width / 2))

        nose_area = background_image[top_left[1]: top_left[1] + nose_width,
                    top_left[0]: top_left[0] + nose_width]

        if nose_area.shape[0] > 0 and nose_area.shape[1] > 0:
            # Original image w/ pig
            nose_pig = cv2.resize(nose_image, (nose_width, nose_width))
            nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
            _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)
            
            # defaults
            nose_mask_cropped = nose_mask
            nose_pig_cropped = nose_pig

            nose_mask_cropped = nose_mask[0:nose_area.shape[0], 0:nose_area.shape[1]]
            nose_pig_cropped = nose_pig[0:nose_area.shape[0], 0:nose_area.shape[1]]

            nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask_cropped)

            final_nose = cv2.add(nose_area_no_nose, nose_pig_cropped)

            background_image[top_left[1]: top_left[1] + nose_width,
                    top_left[0]: top_left[0] + nose_width] = final_nose

def render_mask(results, image, mask_image):
    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        shape = image.shape
        
        top_face = (int(face.landmark[10].x*shape[1]), int(face.landmark[10].y*shape[0]))
        bottom_face = (int(face.landmark[152].x*shape[1]), int(face.landmark[152].y*shape[0]))
        left_face = (int(face.landmark[234].x*shape[1]), int(face.landmark[234].y*shape[0]))
        right_face = (int(face.landmark[454].x*shape[1]), int(face.landmark[454].y*shape[0]))

        face_width = int(hypot(abs(left_face[0]-right_face[0]), abs(left_face[1]-right_face[1])) * 1.3)
        face_height = int(hypot(abs(top_face[0]-bottom_face[0]), abs(top_face[1]-bottom_face[1])) * 1.2)

        center_face = ((top_face[0]+bottom_face[0])//2, (left_face[1]+right_face[1])//2)

        top_left = (int(center_face[0] - face_width / 2), int(center_face[1] - face_height / 2))

        face_area = image[top_left[1]: top_left[1] + face_height,
                    top_left[0]: top_left[0] + face_width]

        if face_area.shape[0] > 0 and face_area.shape[1] > 0:
            # Original image w/ pig
            angle = atan((top_face[0]-bottom_face[0])/(top_face[1]-bottom_face[1])) * (180/pi)

            mask = cv2.resize(mask_image, (face_width, face_height))
            matrix = cv2.getRotationMatrix2D((face_width//2, face_height//2), angle, 1.0)
            rotated_mask = cv2.warpAffine(mask, matrix, (face_width, face_height))

            mask_ = rotated_mask
            mask_gray = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
            _, mask_mask = cv2.threshold(mask_gray, 25, 255, cv2.THRESH_BINARY_INV)
            
            # defaults
            mask_mask_cropped = mask_mask
            mask_cropped = mask_

            mask_mask_cropped = mask_mask[0:face_area.shape[0], 0:face_area.shape[1]]
            mask_cropped = mask_[0:face_area.shape[0], 0:face_area.shape[1]]

            mask_area_no_face = cv2.bitwise_and(face_area, face_area, mask=mask_mask_cropped)

            final_nose = cv2.add(mask_area_no_face, mask_cropped)

            image[top_left[1]: top_left[1] + face_height,
                    top_left[0]: top_left[0] + face_width] = final_nose
