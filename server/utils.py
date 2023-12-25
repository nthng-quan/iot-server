import json

def append_to_file(filename, data):
    file_path = filename

    with open(file_path, 'w') as file:
        file.write(json.dumps(data))

def read_file(filename):
    file_path = filename

    with open(file_path, 'r') as file:
        data = json.load(file)
        return data