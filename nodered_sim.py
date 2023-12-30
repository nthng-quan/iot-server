from flask import Flask, request, jsonify  

app = Flask(__name__)

@app.route('/capture', methods=['POST'])
def capture():
    captured_data = request.get_json()
    print('CAPTURE :', captured_data)
    return captured_data

@app.route('/fire', methods=['POST'])
def fire():
    captured_data = request.get_json()
    print('FIRE :', captured_data)
    return captured_data

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1880)
