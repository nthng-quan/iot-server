from flask import Flask, jsonify, request
import os
import json
from utils import *

app = Flask(__name__)
config = read_file("config.json")
model = FireClassifier(num_classes=2, freeze=True)
img_cfg = config["server"]["image"]
img_server_url = f'{config["server"]["host"]}:{img_cfg["port"]}'
camera_url = config["esp32_cam"]["stream"]

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fire detection API"})

@app.route("/config", methods=["GET", "POST"])
def config():
    config = read_file("config.json")
    if request.method == "POST":
        if request:
            data = request.get_json()
            config["iot_device"] = data
            append_to_file("config.json", config)
            return jsonify({"message": "Config updated"})
        else:
            return jsonify({"message": "No update"})
    else:
        if request.user_agent.string.lower() == "esp8266httpclient":
            return jsonify(config["iot_device"])
        return config

@app.route("/system", methods=["GET", "POST"])
def upload():   
    if request.method == "POST":
        if request:
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
            data = request.get_json()
            reponse = jsonify({
                "time" : formatted_time,
                "data": data
            })
            append_to_file("data.json", reponse.json)
            log_data(data, "./log/system.csv")
            return jsonify({"message": "Upload ok"})
        else:
            return jsonify({"message": "Nothing uploaded"})
    if request.method == "GET":
        data = read_file("data.json")
        return jsonify(data)

@app.route("/fire", methods=["GET", "POST"])
def fire():
    if request.method == "POST":
        if request.data.decode('utf-8') == "check":
            print(request.data)
            return jsonify({"status": "ok"})
        else:
            system_data = request.get_json()
            img_fn = f"{img_cfg['path']}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.jpg"
            img_url = f"http://{img_server_url}/{img_fn.replace('../', '')}"
            status = capture_image(f"http://{camera_url}", img_fn)
            result = model.predict(img_fn)

            log_data(system_data, "./log/fire.csv", result, img_url)

            if status == -1:
                return jsonify({"message": "Error capturing image"})
            else:
                return jsonify({
                    "fire": result, 
                    "url": img_url
                })
    else:
        result = get_fire_report("./log/fire.csv")
        return result

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5555)