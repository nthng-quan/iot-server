import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

from utils.model import ModelResponse


def update_file(file_path: str, data: Dict[str, Any]):
    with open(file_path, "w") as filename:
        json.dump(data, filename, sort_keys=True, indent=4)


def read_file(file_path: str):
    with open(file_path, "r") as file:
        data = json.load(file)
        return data


def log_data(
    log_filename: str,
    system_data: Dict[str, Any],
    model_response: ModelResponse = {"fire": False, "url": ""},
) -> None:
    df_ir = pd.DataFrame([system_data.get("IR", {})], columns=["IR"])
    df_mq135 = pd.DataFrame([system_data.get("MQ_135", {})])
    df_dht = pd.DataFrame([system_data.get("DHT", {})])
    df_servo = pd.DataFrame([system_data.get("servo", {})])

    result_df = pd.DataFrame(
        [datetime.now().strftime("%Y-%m-%d-%H-%M-%S")], columns=["time"]
    )
    result_df = pd.concat([result_df, df_ir, df_mq135, df_dht, df_servo], axis=1)

    if model_response["url"] != "":
        result_df["result"] = model_response["fire"]
        result_df["img_url"] = model_response["url"]

    if not os.path.isfile(log_filename):
        result_df.to_csv(log_filename, header=True, index=False)
    else:
        result_df.to_csv(log_filename, mode="a", header=False, index=False)


def get_fire_report(log_filename: str) -> Dict[str, Any]:
    df = pd.read_csv(log_filename)
    fire = len(df[df["result"] == 1])
    no_fire = len(df[df["result"] == 0])
    lastest_fire = df[df["result"] == 1].tail(1)

    return {
        "n_fire": fire,
        "n_no_fire": no_fire,
        "lastest_fire": lastest_fire.to_dict(orient="records"),
    }
