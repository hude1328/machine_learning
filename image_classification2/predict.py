import base64

import requests
import torch
from PIL import Image
from redis import StrictRedis
import json
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable


def pred(img):
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

    model.eval()
    preds = model(img_variable)

    # convert the pred result with labels
    content = requests.get("https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json").text
    labels = json.loads(content)
    percentage = torch.nn.functional.softmax(preds, dim=1)[0]
    predictions = []
    for i, score in enumerate(percentage.data.numpy()):
        predictions.append((score, labels[str(i)][1]))

    predictions.sort(reverse=True)

    # store the pred result with a dict
    pred_result = []
    for i in range(5):
        pred_result.append({'label': predictions[i][1], 'score': ("%.4f" % predictions[i][0])})
    return pred_result



if __name__ == '__main__':
    queue = StrictRedis(host='localhost', port=6379)
    model = models.inception_v3(pretrained=True)
    model.transform_input = True

    while True:
        ##receive msg
        data = queue.blpop("image")
        json_dict = data[1].decode('utf-8')
        data_dict = json.loads(json_dict)
        URL = data_dict['url']
        timestamp = data_dict['timestamp']

        ##decode the image data
        encoded_image_str = data_dict['image']
        encoded_image = encoded_image_str.encode('utf-8')
        img_data = base64.b64decode(encoded_image)
        with open('img.png', 'wb') as outfile:
            outfile.write(img_data)

        ##feed the image to preloaded V3 model to generate prediction
        image = Image.open('img.png')
        pred_result = pred(image)
        pred_result_json = json.dumps(pred_result)

        ##send back the prediction result
        result_dict = {
            'predictions': pred_result_json,
            'url': URL,
            'timestamp': timestamp
        }
        json_result = json.dumps(result_dict)
        queue.rpush("prediction", json_result)






