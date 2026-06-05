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



def check_eyes_visibility(frame, landmarks, w, h):
    """
    Göz çevresindeki piksel yoğunluğunu ve engelleri (gözlük, saç, kötü ışık vb.) 
    kontrol ederek gözlerin net görünüp görünmediğini analiz eder.
    """
    
    left_x = int(landmarks[33].x * w)
    right_x = int(landmarks[263].x * w)
    
    top_y = int(min(landmarks[159].y, landmarks[386].y) * h)
    bottom_y = int(max(landmarks[145].y, landmarks[374].y) * h)
    
    eye_width = right_x - left_x
    pad = int(eye_width * 0.25) 

    y1 = max(0, top_y - pad)
    y2 = min(h, bottom_y + pad)
    x1 = max(0, left_x)
    x2 = min(w, right_x)
    
    roi = frame[y1:y2, x1:x2]
    
    
    if roi is None or roi.shape[0] < 5 or roi.shape[1] < 5:
        return False
        
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    
    edges = cv2.Canny(blurred, 80, 200)
    
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    
    if edge_density > 0.15: 
        return True
    return False


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
ALARM_TIME_THRESHOLD = 3.0  
HEAD_TILT_THRESHOLD = 20.0  
TILT_TIME_THRESHOLD = 3.0   

closed_start = None
tilt_start = None
ZOOM_FACTOR = 1.5  

current_center_x = None
current_center_y = None

eyes_obscured = False
warning_approved = False


obscured_frames_count = 0
OBSCURED_FRAMES_THRESHOLD = 30 


while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    target_center_x, target_center_y = w // 2, h // 2

    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    EAR = None
    angle_deg = None

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        
        
        if not eyes_obscured:
            if check_eyes_visibility(frame, landmarks, w, h):
                obscured_frames_count += 1
                if obscured_frames_count >= OBSCURED_FRAMES_THRESHOLD:
                    eyes_obscured = True
            else:
                
                obscured_frames_count = 0 

        
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        target_center_x = int((min(xs) + max(xs)) / 2)
        target_center_y = int((min(ys) + max(ys)) / 2)

        
        for landmark in landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)


        leftEAR = calculate_EAR(landmarks, LEFT_EYE)
        rightEAR = calculate_EAR(landmarks, RIGHT_EYE)
        EAR = (leftEAR + rightEAR) / 2.0
        
        
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]
        dx = (right_eye_outer.x - left_eye_outer.x) * w
        dy = (right_eye_outer.y - left_eye_outer.y) * h
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)


    
    if current_center_x is None:
        current_center_x, current_center_y = target_center_x, target_center_y
    else:
        smoothness = 0.1
        current_center_x = int(current_center_x * (1 - smoothness) + target_center_x * smoothness)
        current_center_y = int(current_center_y * (1 - smoothness) + target_center_y * smoothness)

    radius_x = int(w / (2 * ZOOM_FACTOR))
    radius_y = int(h / (2 * ZOOM_FACTOR))

    x1 = current_center_x - radius_x
    y1 = current_center_y - radius_y
    x2 = current_center_x + radius_x
    y2 = current_center_y + radius_y

    if x1 < 0:
        x1, x2 = 0, 2 * radius_x
    if x2 > w:
        x2, x1 = w, w - 2 * radius_x
    if y1 < 0:
        y1, y2 = 0, 2 * radius_y
    if y2 > h:
        y2, y1 = h, h - 2 * radius_y

    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)

    cropped_frame = frame[y1:y2, x1:x2]
    output_frame = cv2.resize(cropped_frame, (w, h))

    
    
    
    if eyes_obscured and not warning_approved:
        overlay = output_frame.copy()
        cv2.rectangle(overlay, (20, h//2 - 60), (w-20, h//2 + 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.85, output_frame, 0.15, 0, output_frame)
        
        cv2.putText(output_frame, "DIKKAT: Sistem gozleri tam olarak goremiyor!", (40, h//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(output_frame, "Bu durum sistemin tam olarak calismamasina neden olabilir.", (40, h//2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(output_frame, "Onaylayip devam etmek icin 'E' harfine basin.", (40, h//2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    
    elif EAR is not None and angle_deg is not None:
        
        
        if EAR < EAR_THRESHOLD:
            if closed_start is None:
                closed_start = time.time()
            ear_elapsed = time.time() - closed_start
            cv2.putText(output_frame, f"GOZ KAPALI: {round(ear_elapsed,1)} sn",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            closed_start = None
            ear_elapsed = 0.0
            cv2.putText(output_frame, "GOZ ACIK",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        
        if abs(angle_deg) > HEAD_TILT_THRESHOLD:
            if tilt_start is None:
                tilt_start = time.time()
            tilt_elapsed = time.time() - tilt_start
            cv2.putText(output_frame, f"KAFA EGIK: {round(tilt_elapsed,1)} sn",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        else:
            tilt_start = None
            tilt_elapsed = 0.0
            cv2.putText(output_frame, "KAFA DIK",
                        (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        
        cv2.putText(output_frame, f"Aci: {round(angle_deg, 1)}",
                    (30, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        
        if ear_elapsed >= ALARM_TIME_THRESHOLD or tilt_elapsed >= TILT_TIME_THRESHOLD:
            if not alarm_active:
                alarm_active = True
                threading.Thread(target=alarm, daemon=True).start()
        else:
            alarm_active = False

    cv2.imshow("Uyku Tespit Sistemi", output_frame)


    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    
    
    elif key == ord('e') or key == ord('E'):
        if eyes_obscured and not warning_approved:
            warning_approved = True
            eyes_obscured = False 
            warning_approved = False
            obscured_frames_count = 0

cap.release()
cv2.destroyAllWindows()