import os
import json
from utils import *
from flask import (
    Flask,
    jsonify,
    request,
    send_from_directory,
    redirect,
    url_for
)

app = Flask(__name__)

cfg = read_file("config.json")
server_cfg = cfg["server"]
server_url = f"http://{server_cfg['host']}:{server_cfg['port']}"    
node_red_url = f"http://{cfg['node_red']['host']}:{cfg['node_red']['port']}"
camera_url = f"http://{cfg['esp32_cam']['host']}"

model = Model(model_path=server_cfg['model'], model_name='yolo')

@app.route("/", methods=["GET"])
def home():
    return jsonify({"service": "Fire detection system API"})


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"message": "ok"})


@app.route("/config", methods=["GET", "POST"])
def config_route():
    config = read_file("config.json")
    if request.method == "POST":
        if request:
            data = request.get_json()
            config["iot_device"] = data
            update_file("config.json", config)
            return jsonify({"message": "Config updated"})
        else:
            return jsonify({"message": "No update"})
    else:
        if request.user_agent.string.lower() == "esp8266httpclient":
            return jsonify(config["iot_device"])
        return config


@app.route("/system", methods=["GET", "POST"])
def system():
    if request.method == "POST":
        if request:
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
            data = request.get_json()
            reponse = jsonify({
                "time" : formatted_time,
                "data": data
            })
            update_file("data.json", reponse.json)
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
            print("* Init check fire")
            return jsonify({"status": "ok"})
        else:
            system_data = request.get_json()
            img_dir, img_url = capture_image()
            result = model.predict(img_dir)
            
            if len(result) > 1: # yolo bboxes
                img_url = result[1]
                result = result[0]

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
    img_path_plain = os.path.join(f"{server_cfg['image_dir']}/plain", filename)
    img_path_yolo = os.path.join(f"{server_cfg['image_dir']}/yolo_output", filename)

    if not os.path.isfile(img_path_plain) and not os.path.isfile(img_path_yolo):
        return jsonify({"error": f"Image {filename} not found"})
    else:
        try:
            return send_from_directory(\
                f"{server_cfg['image_dir']}/plain", filename, as_attachment=True)
        except:
            return send_from_directory(\
                f"{server_cfg['image_dir']}/yolo_output", filename, as_attachment=True)

@app.route("/capture", methods=["GET"])
def capture():
    img_dir, img_url = capture_image()
    if request.user_agent.string.lower() == "esp8266httpclient":
        requests.post(f"{node_red_url}/capture", json={"img_url": img_url})
        return jsonify({"success": "Forwarded to node-red"})

    if img_dir == -1:
        return jsonify({"error": "Error capturing image"})
    else:
        return jsonify({"img_url": img_url})


@app.route("/config/camera", methods=["GET", "POST"])
def camera_config():
    if request.method == "POST":
        if request:
            data = request.get_json()
            result = set_camera_parameters(data)

            if result == 1:
                update_camera_cfg() # get /status -> update config.json
                return jsonify({"message": "Config updated"})
            else:
                return jsonify({"message": result})

        else:
            return jsonify({"message": "No update"})

    else:
        return redirect(url_for("config_route"))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5555)
