import json
import os
from datetime import datetime
import requests
from .helper import read_file, update_file

def capture_image():
    config = read_file("config.json")
    server_url = f'http://{config["server"]["host"]}:{config["server"]["port"]}'
    camera_url = f'http://{config["esp32_cam"]["host"]}/capture'
    
    img_fn = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jpg"
    img_dir = f"{config['server']['image_dir']}/plain/{img_fn}"
    img_url = f"{server_url}/image/{img_fn}"
    # img_dir = f"{config['server']['image_dir']}/test.jpg"
    # img_url = f"{server_url}/image/test.jpg"

    # led on at 18 -> 6
    if datetime.now().hour < 6 or datetime.now().hour > 17:
        set_camera_parameters({"led_intensity": 30})

    try:
        response = requests.get(camera_url)
        response.raise_for_status()

        with open(img_dir, "wb") as file:
            file.write(response.content)

        print(f"-> Image captured and saved to {img_dir}.")
        return img_dir, img_url
    
    except: # test
        img_dir = './log/images/plain/test.jpg'
        img_url = f"{server_url}/image/test.jpg"
        return img_dir, img_url

    # except requests.exceptions.RequestException as e:
    #     print(f"Error: {e}")
    #     return -1, None

def update_camera_cfg():
    config = read_file("config.json")
    current_config = requests.get(f'http://{config["esp32_cam"]["host"]}/status')
    current_config = current_config.json()

    config["esp32_cam"]["detail"] = current_config
    update_file("config.json", config)


def set_camera_parameters(parameters):
    # framesize, quality, gainceling, led_intensity
    config = read_file("config.json")
    base_url = f'http://{config["esp32_cam"]["host"]}/control'

    try:
        for variable, value in parameters.items():
            request_url = f'{base_url}?var={variable}&val={value}'
            response = requests.get(request_url)
            
            if response.status_code == 200:
                print(f'Set {variable} to {value}')
            else:
                return(f'Failed to set {variable} to {value}')

    except Exception as e:
        print(f'Error: {e}')
        return 0

    return 1