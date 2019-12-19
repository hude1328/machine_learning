import base64
import time
from io import BytesIO

from PIL import Image
import json

import requests
import socket
from queue import Queue
from threading import Thread


import telepot
from telepot.loop import MessageLoop


##thread1/main-thread for receive msg from bot by user input and put it into queue
def receive_msg(msg):
    content_type, chat_type, chat_id = telepot.glance(msg)
	
	##user input can either be a URL of a picture or a photo
    if content_type == "text":
        image_url = msg["text"]
        image_data = requests.get(image_url).content
        file_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + ".png"
        with open(file_name, 'wb') as outfile:
            outfile.write(image_data)
        dic_send = dict()
        dic_send["file_name"] = file_name
        dic_send["chat_id"] = chat_id
        q1.put(dic_send)
	
    if content_type == 'photo':
        file_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + ".png"
        bot.download_file(msg['photo'][-1]['file_id'], file_name)
        dic_send = dict()
        dic_send["file_name"] = file_name
        dic_send["chat_id"] = chat_id
        q1.put(dic_send)

		
##thread2 for get msg from queue, transcode the msg and send to server through socket, and receive result from server  
def connect_with_server():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('localhost', 50011))
    dic_send = q1.get()
    image = Image.open(dic_send['file_name'])
    chat_id = dic_send['chat_id']

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_img = base64.b64encode(buffered.getvalue())
    encoded_img_str = encoded_img.decode('utf-8')

    data = {
        'image': encoded_img_str,
        'chat_id': chat_id
    }
    json_data = json.dumps(data)
    client_socket.send(json_data.encode('utf-8'))

    data = client_socket.recv(10240000)
    q2.put(data)
    client_socket.close()


##thread3 sending back result to user
def send_response_to_user():
    data = q2.get()
    result_dic = json.loads(data.decode('utf-8'))
    pred_result_json = result_dic['predictions']
    pred_result = json.loads(pred_result_json)

    chat_id = result_dic['chat_id']
    resp = ''
    for i in range(5):
        resp = resp + str(i+1) + '. ' + pred_result[i]['label'] + ' (' + pred_result[0]['proba'] + ')' + '\n'

    bot.sendMessage(chat_id, resp)

if __name__ == '__main__':
    q1 = Queue()
    q2 = Queue()
    bot = telepot.Bot("832061733:AAG484FATrqbU0ggbkl2rCMCKO5vJ7NWgfk")
    MessageLoop(bot, receive_msg).run_as_thread()

    while True:
        t1 = Thread(target=connect_with_server)
        t2 = Thread(target=send_response_to_user)
        t1.start()
        t2.start()
        time.sleep(10)



