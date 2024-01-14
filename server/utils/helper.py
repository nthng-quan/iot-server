import json
import os 
import pandas as pd
from datetime import datetime
import requests

def update_file(file_path, data):
    with open(file_path, 'w') as filename:
        json.dump(data, filename, sort_keys=True, indent=4)


def read_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data


def log_data(system_data, log_fn, result="", img_url=""):
    df_ir = pd.DataFrame([system_data.get('IR', {})], columns=['IR'])
    df_mq135 = pd.DataFrame([system_data.get('MQ_135', {})])
    df_dht = pd.DataFrame([system_data.get('DHT', {})])
    df_servo = pd.DataFrame([system_data.get('servo', {})])

    result_df = pd.DataFrame([datetime.now().strftime('%Y-%m-%d-%H-%M-%S')], columns=['time'])
    result_df = pd.concat([result_df, df_ir, df_mq135, df_dht, df_servo], axis=1)

    if img_url != None and result != None:
        result_df["result"] = result
        result_df["img_url"] = img_url

    if not os.path.isfile(log_fn):
        result_df.to_csv(log_fn, header=True, index=False)
    else:
        result_df.to_csv(log_fn, mode='a', header=False, index=False)


def get_fire_report(log_fn):
    df = pd.read_csv(log_fn)
    fire = len(df[df['result'] == 1])
    no_fire = len(df[df['result'] == 0])
    lastest_fire = df[df['result'] == 1].tail(1)

    return {
        "n_fire": fire,
        "n_no_fire": no_fire,
        "lastest_fire": lastest_fire.to_dict(orient='records')
    }

    