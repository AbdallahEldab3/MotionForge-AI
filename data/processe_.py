import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2 as cv
import numpy as np

count, success = 0, True
model_path = r"C:\Users\IBRA\project_maham\models\pose_landmarker_full.task"
input_video = r"\raw_videos\walk1.mp4"
cap = cv.VideoCapture(input_video)
fps = cap.get(cv.CAP_PROP_FPS)
keypoints = open(r"C:\Users\IBRA\project_maham\data\keypoints\walk1.txt", 'w')
extracted_points_list = []

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO)

with PoseLandmarker.create_from_options(options) as landmarker:

    while success:
        success, frame = cap.read()
        if success:
            current_frame= int(cap.get(cv.CAP_PROP_POS_FRAMES))
            frame_timestamp_ms = (current_frame / fps) * 1000
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            for landmark in pose_landmarker_result.pose_landmarks:
                extracted_points_list.append({
                    'X' : landmark.x,
                    'Y' : landmark.y,
                    'Z' : landmark.z,
                    'visibility' : landmark.visibility
                })
            count +=1
            print(pose_landmarker_result.pose_landmarks[0])
    cap.release()
    
for each in extracted_points_list :
    print(extracted_points_list[each])
    

keypoints.write('extracted_points_list')