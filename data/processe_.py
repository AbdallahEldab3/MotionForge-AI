import mediapipe as mp
import cv2 as cv
import numpy as np
import os

def extract_motion_tensor(video_path, model_path, output_path):
    
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv.CAP_PROP_FPS)
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"FPS: {fps}")
    
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO
    )
    
    motion_frames = []
    frame_idx = 0
    
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((frame_idx / fps) * 1000) if fps > 0 else 0
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            frame_tensor = np.zeros((33, 4), dtype=np.float32)
            
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                for j in range(33):
                    if j < len(landmarks):
                        frame_tensor[j] = [landmarks[j].x, landmarks[j].y, landmarks[j].z, landmarks[j].visibility]
                        frame_tensor[:, 0:2] = np.clip(frame_tensor[:, 0:2], 0.0, 1.0)
            motion_frames.append(frame_tensor)
            frame_idx += 1
    
    cap.release()
    motion_tensor = np.stack(motion_frames, axis=0)  
    np.save(output_path, motion_tensor)
    print("Joint 0 at frame 0:", motion_tensor[0, 0]) #check
    return motion_tensor


if __name__ == "__main__":
    model_path = r"C:\Users\IBRA\project_maham\models\pose_landmarker_heavy.task"
    video_path = r"C:\Users\IBRA\project_maham\data\raw_videos\walk1.mp4"
    output_path = r"C:\Users\IBRA\project_maham\data\keypoints\walk1_motion.npy"
    
    tensor = extract_motion_tensor(video_path, model_path, output_path)
    print(np.any(np.all(tensor == 0, axis=(1,2)))) #check
    print(tensor[..., :2].min(), tensor[..., :2].max()) #check
    print(np.mean(np.abs(tensor[1:] - tensor[:-1]))) #check
    