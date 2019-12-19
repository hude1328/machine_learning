## deploy the pre-trained text classifier as a chatbot on Telegram to allow other people to use it
import telepot
bot = telepot.Bot('832061733:AAG484FATrqbU0ggbkl2rCMCKO5vJ7NWgfk')
bot.getMe()


import time
import telepot
from telepot.loop import MessageLoop
from sklearn.externals import joblib

def handle(msg):
    """
    A function that will be invoked when a message is
    recevied by the bot
    """
    content_type, chat_type, chat_id = telepot.glance(msg)

    if content_type == "text":
        content = msg["text"]
        str_output = " "
        txt = []
        txt.append(content)
        str_output = pred(txt)
        bot.sendMessage(chat_id, str_output)

def pred(text):
    model = joblib.load("model.pkl")
    pred = model.predict(text)
    prob = model.predict_proba(text)
    result = pred[0]
    result_proba_neg = "%.2f" % prob[0][0]
    result_proba_pos = "%.2f" % prob[0][1]
    if result == 0:
        str_output = "This is a negative review! (" + result_proba_neg + ")"
    else:
        str_output = "This is a positive review! (" + result_proba_pos + ")"
        
    return str_output
        
if __name__ == "__main__":
    bot = telepot.Bot("832061733:AAG484FATrqbU0ggbkl2rCMCKO5vJ7NWgfk")   
    MessageLoop(bot, handle).run_as_thread()
    while True:
        time.sleep(10)