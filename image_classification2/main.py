import time
from threading import Thread

from redis import StrictRedis
import validators
import json


def receive_msg():
    URL = input("Please input the URL of the image: ")
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    ##use validators to validate the input message, True:continue, False:restart
    valid = validators.url(URL)
    if valid:
        data_dict ={
            'timestamp': timestamp,
            'url': URL
        }
        data = json.dumps(data_dict)
        queue.rpush("download", data)
    else:
        receive_msg()

def show_result():
    data = queue.blpop("prediction")
    json_dict = data[1].decode('utf-8')
    data_dict = json.loads(json_dict)
    timestamp = data_dict['timestamp']
    URL = data_dict['url']

    ##get and format the prediction result for presentation
    pred_result_json = data_dict['predictions']
    pred_result = json.loads(pred_result_json)

    resp = 'Results: ' + '\n'
    resp += "URL: " + URL + '\n'
    for i in range(5):
        resp = resp + str(i + 1) + '. ' + pred_result[i]['label'] + ' (' + pred_result[0]['score'] + ')' + '\n'
    print(resp)


if __name__ == '__main__':
    queue = StrictRedis(host='localhost', port=6379)

    while True:
        t1 = Thread(target=receive_msg())
        t2 = Thread(target=show_result())
        t1.start()
        t2.start()


