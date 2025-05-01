import os

import utils
from flask import (
    Flask,
    jsonify,
    request,
    send_from_directory,
    redirect,
    url_for,
)
from flask.wrappers import Response

from datetime import datetime
import requests

app = Flask(__name__)

cfg = utils.read_file("config.json")
server_cfg = cfg["server"]
server_url = f"http://{server_cfg['host']}:{server_cfg['port']}"
node_red_url = f"http://{cfg['node_red']['host']}:{cfg['node_red']['port']}"
camera_url = f"http://{cfg['esp32_cam']['host']}"

model = utils.Model(model_path=server_cfg["model"], model_name="yolo")


@app.route("/", methods=["GET"])
def home():
    return jsonify({"service": "Fire detection system API"})


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"message": "ok"})


@app.route("/config", methods=["GET"])
def get_config():
    config = utils.read_file("config.json")
    if request.user_agent.string.lower() == "esp8266httpclient":
        return jsonify(config.get("iot_device", {}))
    return config


@app.route("/config", methods=["POST"])
def update_config():
    data = request.get_json()

    config = utils.read_file("config.json")
    config["iot_device"] = data
    utils.update_file("config.json", config)
    return jsonify({"message": "Config updated"})


@app.route("/system", methods=["GET"])
def get_system_data():
    data = utils.read_file("data.json")
    return jsonify(data)


@app.route("/system", methods=["POST"])
def post_system_data():
    data = request.get_json()
    if not data:
        return jsonify({"message": "Nothing uploaded"}), 400

    formatted_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    response_data = {"time": formatted_time, "data": data}

    utils.update_file("data.json", response_data)
    utils.log_data(data, "./log/system.csv")

    return jsonify({"message": "Upload ok"})


@app.route("/fire", methods=["GET"])
def get_fire_report():
    return utils.get_fire_report("./log/fire.csv")


@app.route("/fire", methods=["POST"])
def check_or_detect_fire():
    # Handle ESP32 "check"
    if request.data.decode("utf-8") == "check":
        print("* Init check fire")
        return jsonify({"status": "ok"})

    # Handle actual fire detection
    system_data = request.get_json()
    img_dir, img_url = utils.capture_image()

    if img_dir == -1:
        return jsonify({"error": "Error capturing image"}), 500

    result = model.predict(img_dir)

    if len(result) > 1:
        img_url = result[1]
        result = result[0]

    utils.log_data(system_data, "./log/fire.csv", result, img_url)

    final_response = {"fire": result, "url": img_url}
    return jsonify(final_response)


@app.route("/image/<path:filename>", methods=["GET"])
def get_image(filename):
    base_img_dir = server_cfg["image_dir"]
    paths = {
        "plain": os.path.join(base_img_dir, "plain", filename),
        "yolo": os.path.join(base_img_dir, "yolo_output", filename),
    }

    for subdir, path in paths.items():
        if os.path.isfile(path):
            image: Response = send_from_directory(
                os.path.join(base_img_dir, subdir), filename, as_attachment=True
            )
            return image

    return jsonify({"error": f"Image {filename} not found"}), 404


@app.route("/capture", methods=["GET"])
def capture():
    img_dir, img_url = utils.capture_image()
    if request.user_agent.string.lower() == "esp8266httpclient":
        requests.post(f"{node_red_url}/capture", json={"img_url": img_url})
        return jsonify({"success": "Forwarded to node-red"})

    if img_dir == -1:
        final_response = {"error": "Error capturing image"}
        return jsonify(final_response)
    else:
        final_response = {"img_url": img_url}
        return jsonify(final_response)


@app.route("/config/camera", methods=["GET"])
def get_camera_config():
    return redirect(url_for("routes.config_route"))


@app.route("/config/camera", methods=["POST"])
def update_camera_config():
    data = request.get_json()
    if not data:
        result = {"message": "No update"}
        return jsonify(result), 400

    result = utils.set_camera_parameters(data)

    if result == 1:
        utils.update_camera_cfg()  # sync config.json
        final_response = {"message": "Config updated"}
        return jsonify(final_response)
    else:
        final_response = {"message": result}
        return jsonify(final_response)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5555)
