import os
from typing import Any, Dict, Literal, Tuple, TypeAlias, TypedDict, Optional
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

from utils.model import Model, ModelResponse

app = Flask(__name__)

cfg = utils.read_file("config.json")
server_cfg = cfg["server"]
server_url = f"http://{server_cfg['host']}:{server_cfg['port']}"
node_red_url = f"http://{cfg['node_red']['host']}:{cfg['node_red']['port']}"
camera_url = f"http://{cfg['esp32_cam']['host']}"

model = Model(model_path=server_cfg["model"], model_name="yolo")

SystemRouteResponse: TypeAlias = Tuple[Response, Optional[int]]


class ResponseMessage(TypedDict):
    status: Literal["success", "error"]
    message: Optional[str | Dict[str, Any] | ModelResponse | Exception]


@app.route("/", methods=["GET"])
def home() -> SystemRouteResponse:
    return jsonify(
        ResponseMessage(status="success", message="Fire detection system API")
    ), 200


@app.route("/status", methods=["GET"])
def status() -> SystemRouteResponse:
    return jsonify(ResponseMessage(status="success", message="OK")), 200


@app.route("/config", methods=["GET"])
def get_config() -> SystemRouteResponse:
    config = utils.read_file("config.json")

    if request.user_agent.string.lower() == "esp8266httpclient":
        config = config.get("iot_device", {})

    return jsonify(ResponseMessage(status="success", message=config)), 200


@app.route("/config", methods=["POST"])
def update_config() -> SystemRouteResponse:
    data = request.get_json()

    config = utils.read_file("config.json")
    config["iot_device"] = data
    try:
        utils.update_file("config.json", config)
        return jsonify(ResponseMessage(status="success", message="Config updated")), 200
    except Exception as e:
        return jsonify(ResponseMessage(status="success", message=e)), 400


@app.route("/system", methods=["GET"])
def get_system_data() -> SystemRouteResponse:
    data = utils.read_file("data.json")
    return jsonify(ResponseMessage(status="success", message=data)), 200


@app.route("/system", methods=["POST"])
def post_system_data() -> SystemRouteResponse:
    data = request.get_json()
    if not data:
        return jsonify(ResponseMessage(status="error", message="Nothing uploaded")), 400

    formatted_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    response_data = {"time": formatted_time, "data": data}

    utils.update_file("data.json", response_data)
    utils.log_data(log_filename="./log/system.csv", system_data=data)

    return jsonify(ResponseMessage(status="success", message="Upload ok")), 200


@app.route("/fire", methods=["GET"])
def get_fire_report() -> SystemRouteResponse:
    try:
        fire_report = utils.get_fire_report("./log/fire.csv")
        return jsonify(ResponseMessage(status="success", message=fire_report)), 200
    except Exception as e:
        return jsonify(ResponseMessage(status="error", message=e)), 400


@app.route("/fire", methods=["POST"])
def check_or_detect_fire() -> SystemRouteResponse:
    # Handle ESP32 "check"
    if request.data.decode("utf-8") == "check":
        print("* Init check fire")
        return jsonify(
            ResponseMessage(status="success", message="Initalize check fire"), 200
        )

    # Handle actual fire detection
    system_data: Dict[str, Any] = request.get_json()
    img_info: Dict[str, str] = utils.capture_image()

    if img_info.get("img_dir", "") == "":
        return jsonify(
            ResponseMessage(status="error", message="Error capturing image"), 500
        )

    result: ModelResponse = model.predict(img_info)
    utils.log_data(
        log_filename="./log/fire.csv", system_data=system_data, model_response=result
    )
    return jsonify(ResponseMessage(status="success", message=result)), 200


# TODO: separate service
@app.route("/image/<path:filename>", methods=["GET"])
def get_image(filename) -> Response | SystemRouteResponse:
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

    return jsonify(
        ResponseMessage(status="error", message=f"Image {filename} not found")
    ), 404


@app.route("/capture", methods=["GET"])
def capture() -> SystemRouteResponse:
    img_dir, img_url = utils.capture_image()
    if request.user_agent.string.lower() == "esp8266httpclient":
        requests.post(f"{node_red_url}/capture", json={"img_url": img_url})
        return jsonify(
            ResponseMessage(status="success", message="Forwarded to node-red")
        ), 200

    if img_dir == -1:
        return jsonify(
            ResponseMessage(status="error", message="Error capturing image")
        ), 200
    else:
        return jsonify(
            ResponseMessage(status="success", message={"img_url": img_url})
        ), 200


@app.route("/config/camera", methods=["GET"])
def get_camera_config() -> Response:
    return redirect(url_for("routes.config_route"))


@app.route("/config/camera", methods=["POST"])
def update_camera_config() -> SystemRouteResponse:
    data = request.get_json()
    if not data:
        return jsonify(ResponseMessage(status="success", message="No update")), 400

    result = utils.set_camera_parameters(data)

    if result == 1:
        utils.update_camera_cfg()  # sync config.json
        return jsonify(ResponseMessage(status="success", message="Config updated")), 200
    else:
        return jsonify(ResponseMessage(status="success", message=result)), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5555)
