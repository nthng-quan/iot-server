from flask import Flask, jsonify, request
from datetime import datetime
import os
import json
import torch
import numpy as np
from PIL import Image

from utils import *

app = Flask(__name__)
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

img_server_url = "192.168.1.5:5551"
class FireClassifier(nn.Module):
    def __init__(self, num_classes, freeze=True):
        super().__init__()
        # self.model = models.efficientnet_b0()
        self.model = models.resnet18()

        if freeze:
            for param in self.model.parameters():
                param.require_grad = False
        
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)

model = FireClassifier(num_classes=2, freeze=True)
model.to(device)

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

@app.route("/upload", methods=["GET", "POST"])
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
            return jsonify({"message": "Upload ok"})
        else:
            return jsonify({"message": "Nothing uploaded"})
    else:
        return jsonify({"message": "Upload sensor data"})

@app.route("/get", methods=["GET"])
def get():
    if request.method == "GET":
        if request:
            data = read_file("data.json")
            return jsonify(data)
        else:
            return jsonify({"message": "No data found"})
    else:
        return jsonify({"message": "Get sensor data"})

@app.route("/fire", methods=["GET", "POST"])
def fire():
    if request.method == "POST":
        if request.data.decode('utf-8') == "check":
            print(request.data)
            return jsonify({"status": "ok"})
        else:
            print(request.data)
            return jsonify({"status": "huh"})
    else:
        img = Image.open("ok.jpg").convert('RGB')
        img = data_transform(img)
        img = img.to(device)

        result =  model(img.unsqueeze(0))
        result = torch.argmax(result, dim=1)

        print(result.item())

        return jsonify({
            # "fire": result.item(), 
            # "fire": 0,
            "fire": np.random.randint(0, 2), 
            "url": f"{img_server_url}/ok.jpg"
        })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5555)