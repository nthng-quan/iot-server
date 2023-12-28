# iot-server
Fire detection server

# /system
GET:
```json
{
    "data": {
        "DHT": {
            "humidity": 72,
            "temperature": 29
        },
        "IR": 1,
        "MQ-135": {
            "correctedPPM": 29461.95,
            "correctedRZero": 16.45,
            "ppm": 39623.91,
            "resistance": 9.3,
            "rzero": 14.66
        }
    },
    "time": "2023-12-29-01-37-50"
}
```
POST: arduino

# /fire
GET:
```json
{
    "_fire": 2,
    "_no_fire": 0,
    "lastest_fire": [{
        "IR": 0,
        "base_pos": 0,
        "correctedPPM": 33395.1,
        "correctedRZero": 15.72,
        "humidity": 71,
        "img_url": "http://192.168.1.5:5551/images/2023-12-29-01-07-03.jpg",
        "neck_pos": 0,
        "ppm": 45134.14,
        "resistance": 8.91,
        "result": 1,
        "rzero": 14.1,
        "temperature": 29,
        "time": "2023-12-29-01-07-04"
    }]
}
```
POST: arduino

Notification payload -> NODE-RED: https://github.com/nthng-quan/iot-system/blob/4cf9836fa445fb6023611c7f241686b00c5f8875/wifi_http.cpp#L147C1-L147C1

# /config
GET:
```json
{
    "esp32_cam": {
        "host": "192.168.1.67",
        "stream": "192.168.1.67:81/stream"
    },
    "iot_device": {
        "corrected_ppm": 100000000,
        "corrected_rzero": 0,
        "humidity": 0,
        "servo_base": 90,
        "servo_neck": 0,
        "temperature": 100
    },
    "server": {
        "host": "192.168.1.5",
        "image": {
            "path": "../images",
            "port": 5551
        },
        "main": {
            "ml_checkpoint": "./model/model.pth",
            "port": 5555
        }
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