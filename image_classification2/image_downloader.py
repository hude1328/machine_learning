import base64
from io import BytesIO

import requests
from redis import StrictRedis
import json
from PIL import Image


if __name__ == '__main__':
    queue = StrictRedis(host='localhost', port=6379)
    while True:
        data = queue.blpop("download")
        json_dict = data[1].decode('utf-8')
        data_dict = json.loads(json_dict)
        URL = data_dict['url']
        timestamp = data_dict['timestamp']

        ##Download image from url
        image_data = requests.get(URL).content
        file_name = timestamp + ".png"
        with open(file_name, 'wb') as outfile:
            outfile.write(image_data)

        ##Encode the image with base64
        image = Image.open(file_name)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_img = base64.b64encode(buffered.getvalue())
        encoded_img_str = encoded_img.decode('utf-8')

        data_dict2 = {
            'timestamp': timestamp,
            'url': URL,
            'image': encoded_img_str
        }
        data2 = json.dumps(data_dict2)
        queue.rpush("image", data2)

