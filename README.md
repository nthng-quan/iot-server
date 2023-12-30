# iot-server
Fire detection server

# Setup
```bash
git clone https://github.com/nthng-quan/iot-server.git
cd iot-server/server/
docker compose up
```

# /system
GET:
```json
{
    "data": {
        "DHT": {
            "humidity": 71,
            "temperature": 30
        },
        "IR": 1,
        "MQ_135": {
            "correctedPPM": 14177.37,
            "correctedRZero": 21.33,
            "ppm": 19572.2,
            "resistance": 12.05,
            "rzero": 19.06
        }
    },
    "time": "2023-12-31-03-08-30"
}
```
POST: arduino -> server

https://github.com/nthng-quan/iot-system/blob/4cf9836fa445fb6023611c7f241686b00c5f8875/wifi_http.cpp#L85C13-L85C13

```json
{    
    "DHT": {
        "humidity": 72,
        "temperature": 29
    },
    "IR": 1,
    "MQ_135": {
        "correctedPPM": 29461.95,
        "correctedRZero": 16.45,
        "ppm": 39623.91,
        "resistance": 9.3,
        "rzero": 14.66
    }
}
```

# /fire
GET:
```json
{
    "lastest_fire": [
        {
            "IR": 0,
            "base_pos": 0,
            "correctedPPM": 12028.24,
            "correctedRZero": 22.73,
            "humidity": 71,
            "img_url": "http://192.168.1.5:5555/image/2023_12_31_03_27_27.jpg",
            "neck_pos": 90,
            "ppm": 16424.88,
            "resistance": 12.83,
            "result": 1,
            "rzero": 20.31,
            "temperature": 30,
            "time": "2023-12-31-03-27-27"
        }
    ],
    "n_fire": 2,
    "n_no_fire": 4
}
```
POST: arduino -> server

Arduino payload: https://github.com/nthng-quan/iot-system/blob/4cf9836fa445fb6023611c7f241686b00c5f8875/wifi_http.cpp#L110

```json
{
    "IR": 0,
    "MQ-135": {
        "correctedPPM": 17627.34,
        "correctedRZero": 19.8,
        "ppm": 24070.6,
        "resistance": 11.18,
        "rzero": 17.69
    },
    "DHT": {
        "humidity": 71,
        "temperature": 30
    },
    "servo": {
        "neck_pos": 0,
        "base_pos": 90
    }
}
```

POST: arduino -> NODE-RED

Notification payload: https://github.com/nthng-quan/iot-system/blob/4cf9836fa445fb6023611c7f241686b00c5f8875/wifi_http.cpp#L147C1-L147C1

```json
{
    "IR": 0,
    "MQ_135": {
        "correctedPPM": 17627.34,
        "correctedRZero": 19.8,
        "ppm": 24070.6,
        "resistance": 11.18,
        "rzero": 17.69
    },
    "DHT": {
        "humidity": 71,
        "temperature": 30
    },
    "check_fire": {
        "fire": 1,
        "img_url": "http://192.168.1.5:5555/image/2023-12-29-01-07-03.jpg"
    }
}
```
# /config
GET:
```json
{
    "esp32_cam": {
        "host": "192.168.1.67",
        "stream": "192.168.1.244:81/stream"
    },
    "iot_device": {
        "corrected_ppm": 900000000,
        "corrected_rzero": 0,
        "humidity": 0,
        "servo_base": 90,
        "servo_neck": 90,
        "temperature": 100
    },
    "node_red": {
        "host": "192.168.1.5",
        "port": 1880
    },
    "server": {
        "host": "192.168.1.5",
        "image_dir": "./log/images",
        "model": "./model/",
        "port": 5555
    }
}
```
POST:
sample payload
```json
{
    "servo_neck": 0,
    "servo_base": 90,
    "corrected_ppm": 100000000,
    "corrected_rzero": 0,
    "humidity": 0,
    "temperature": 100
}
``````
# /capture
Arduino <-- GET / resp --> server ~ server -- POST --> node-red             

GET: arduino -> server

Response: https://github.com/nthng-quan/iot-server/blob/10ca4faf19f2ec3540cd7442a28a27aeb0c1e5e6/server/app.py#L110C1-L110C1

POST: server -> node-red
```json
{
    "img_url": "http://192.168.1.5:5555/image/2023-12-29-01-07-03.jpg"
}
```