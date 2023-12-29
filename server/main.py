import os
import json
from utils import *
from flask import (
    Flask,
    jsonify,
    request,
    send_from_directory
)

app = Flask(__name__)
cfg = read_file("config.json")
server_cfg = cfg["server"]
server_url = f'{server_cfg["host"]}:{server_cfg["port"]}'
camera_url = cfg["esp32_cam"]["stream"]

model = Model(model_path=server_cfg['model'], reload=True)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fire detection API"})

@app.route("/health", methods=["GET"])
def healthcheck():
    return jsonify({"message": "ok"})

@app.route("/config", methods=["GET", "POST"])
def config():
    cfg = read_file("config.json")
    if request.method == "POST":
        if request:
            data = request.get_json()
            cfg["iot_device"] = data
            append_to_file("config.json", cfg)
            return jsonify({"message": "Config updated"})
        else:
            return jsonify({"message": "No update"})
    else:
        if request.user_agent.string.lower() == "esp8266httpclient":
            return jsonify(cfg["iot_device"])
        return cfg

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
            img_fn = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jpg"
            img_dir = f"{server_cfg['image_dir']}/{img_fn}"
            img_url = f"http://{server_url}/image/{img_fn}"

            status = capture_image(f"http://{camera_url}", img_fn)
            result = model.predict(img_fn)

            log_data(system_data, "./log/fire.csv", result, img_url)

            if status == -1:
                return jsonify({"error": "Error capturing image"})
            else:
                return jsonify({
                    "fire": result, 
                    "url": img_url
                })
    else:
        result = get_fire_report("./log/fire.csv")
        return result

@app.route("/image/<path:filename>", methods=["GET"])
def get_image(filename):
    img_path = os.path.join(server_cfg['image_dir'], filename)
    if not os.path.isfile(img_path):
        return jsonify({"error": f"Image {img_path} not found"})
    else:
        return send_from_directory(
            server_cfg['image_dir'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5555)
