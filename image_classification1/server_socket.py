import socket
from queue import Queue
from threading import Thread

import requests
import torch
import torchvision.models as models
import base64
import json
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable


def connect_with_client():
    data = client_socket.recv(10240000)
    q.put(data)

def handle():
    data = q.get()
    json_data = data.decode('utf-8')
    receive_dic = json.loads(json_data)
    encoded_img_str = receive_dic['image']
    chat_id = receive_dic['chat_id']
    encoded_img = encoded_img_str.encode('utf-8')
    img_data = base64.b64decode(encoded_img)
    with open('image.png', 'wb') as outfile:
        outfile.write(img_data)

    image = Image.open('image.png')
    pred_result = pred(image)
    pred_result_json = json.dumps(pred_result)
    result_dic = {
        'predictions': pred_result_json,
        'chat_id': chat_id
    }
    json_result = json.dumps(result_dic)

    client_socket.sendall(json_result.encode('utf-8'))
    client_socket.close()

def pred(img):
    model = models.inception_v3(pretrained=True)
    model.transform_input = True

    ##preprocess the input image for later prediction
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img)
    img_tensor.unsqueeze_(0)
    img_variable = Variable(img_tensor)

    #predict
    model.eval()
    preds = model(img_variable)

    #convert the pred result with labels
    content = requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json").text
    labels = json.loads(content)
    percentage = torch.nn.functional.softmax(preds, dim=1)[0]
    predictions = []
    for i, score in enumerate(percentage.data.numpy()):
        predictions.append((score, labels[str(i)][1]))

    predictions.sort(reverse=True)

    #store the pred result with a dict
    pred_result = []
    for i in range(5):
        pred_result.append({'label': predictions[i][1], 'proba': ("%.4f" % predictions[i][0])})
    return pred_result


if __name__ == '__main__':
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 50011))
    server_socket.listen(10)
    while True:
        (client_socket, addr) = server_socket.accept()
        q = Queue()
        t1 = Thread(target=connect_with_client())
        t2 = Thread(target=handle())
        t1.start()
        t2.start()
