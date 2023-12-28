import json
import cv2
import os 
import pandas as pd
from datetime import datetime

def append_to_file(filename, data):
    file_path = filename

    with open(file_path, 'w') as file:
        file.write(json.dumps(data))

def read_file(filename):
    file_path = filename

    with open(file_path, 'r') as file:
        data = json.load(file)
        return data

def log_data(system_data, log_fn, result="", img_url=""):
    df_ir = pd.DataFrame([system_data.get('IR', {})], columns=['IR'])
    df_mq135 = pd.DataFrame([system_data.get('MQ-135', {})])
    df_dht = pd.DataFrame([system_data.get('DHT', {})])
    df_servo = pd.DataFrame([system_data.get('servo', {})])

    result_df = pd.DataFrame([datetime.now().strftime('%Y-%m-%d-%H-%M-%S')], columns=['time'])
    result_df = pd.concat([result_df, df_ir, df_mq135, df_dht, df_servo], axis=1)

    if img_url and result:
        result_df["result"] = result
        result_df["img_url"] = img_url

    if not os.path.isfile(log_fn):
        result_df.to_csv(log_fn, header=True, index=False)
    else:
        result_df.to_csv(log_fn, mode='a', header=False, index=False)

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

def get_fire_report(log_fn):
    df = pd.read_csv(log_fn)
    fire = len(df[df['result'] == 1])
    no_fire = len(df[df['result'] == 0])
    lastest_fire = df[df['result'] == 1].tail(1)

    return {
        "_fire": fire,
        "_no_fire": no_fire,
        "lastest_fire": lastest_fire.to_dict(orient='records')
    }