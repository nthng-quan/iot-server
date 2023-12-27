import json
import cv2

def append_to_file(filename, data):
    file_path = filename

    with open(file_path, 'w') as file:
        file.write(json.dumps(data))

def read_file(filename):
    file_path = filename

    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

def capture_image(video_url, save_path):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return -1
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from the video stream.")
        return -1
        
    cv2.imwrite(save_path, frame)
    cap.release()

    print(f"Image captured and saved to {save_path}")
    return 0