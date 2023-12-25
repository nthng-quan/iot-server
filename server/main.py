from flask import Flask, jsonify, request
from datetime import datetime
import os
import json
import torch

from PIL import Image

# import mlflow
# import mlflow.pytorch
# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("fire-detection")

from utils import *

app = Flask(__name__)
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

server_url = "192.168.1.5:5551"

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

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fire detection API"})


@app.route("/upload", methods=["GET", "POST"])
def upload():   
    if request.method == "POST":
        if request:
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d-%H-%M-%S")
            # data = request.data.decode('utf-8')
            data = request.get_json()
            print(type(data))
            print(data)
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
        print(request.data)
        if request.data.decode('utf-8') == "check":
                return jsonify({"status": "ok"})
        else:
            return jsonify({"status": "huh"})
    else:
        img = Image.open("ok.jpg").convert('RGB')
        img = data_transform(img)

        result =  model(img.unsqueeze(0))
        result = torch.argmax(result, dim=1)

        print(result.item())

        return jsonify({
            "fire": result.item(), 
            "url": f"{server_url}/ok.jpg"
        })


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5555)