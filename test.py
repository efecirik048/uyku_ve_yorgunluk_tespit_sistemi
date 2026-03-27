import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import winsound

from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions





model_path = "face_landmarker.task"

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)




LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]




def calculate_EAR(landmarks, eye_indices):

    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])

    vertical1 = np.linalg.norm(p2 - p6)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)

    return ear




alarm_active = False

def alarm():

    global alarm_active

    while alarm_active:

        winsound.Beep(1000, 500)
        time.sleep(0.5)




cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.20

closed_start = None




while True:

    ret, frame = cap.read()

    if not ret:
        break


    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)


    if result.face_landmarks:

        landmarks = result.face_landmarks[0]


        

        h, w, _ = frame.shape

        for landmark in landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)



        

        leftEAR = calculate_EAR(landmarks, LEFT_EYE)
        rightEAR = calculate_EAR(landmarks, RIGHT_EYE)

        EAR = (leftEAR + rightEAR) / 2.0



        

        if EAR < EAR_THRESHOLD:

            if closed_start is None:

                closed_start = time.time()


            elapsed = time.time() - closed_start


            cv2.putText(frame, f"KAPALI: {round(elapsed,1)} sn",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        2)


            if elapsed >= 3:

                if not alarm_active:

                    alarm_active = True

                    threading.Thread(target=alarm, daemon=True).start()



        

        else:

            closed_start = None

            alarm_active = False



            cv2.putText(frame, "GOZ ACIK",
                        (30,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)




    cv2.imshow("Uyku Tespit Sistemi", frame)



    

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break



cap.release()

cv2.destroyAllWindows()
