import json
import base64
import requests
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', help='Image File Path')
    args = parser.parse_args()
    return args


def get_base64_encoded_string(filename):
    with open(filename, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read())
        encoded_string = encoded_string.decode('utf-8')

    return encoded_string


if __name__ == '__main__':
    args = parse_args()
    img_path = args.image
    img_bytes = get_base64_encoded_string(img_path)

    url = 'http://localhost:5000/generate'
    headers = { 'Content-Type': 'application/json' }
    data = { 'img_bytes': img_bytes }
    response = requests.post(url, headers=headers, json=data)

    response_text = json.loads(response.text)
    print(json.dumps(response_text, indent=2, ensure_ascii=False))
