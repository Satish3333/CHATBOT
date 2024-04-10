from flask import*

import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
#colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

app = Flask(__name__)
@app.route('/')
@app.route('/res',methods=['POST','GET'])
def res():
    if request.method=="POST":
        inp=request.form['name']
        with open(r'einstein3.json') as file:
            data = json.load(file)
        model = keras.models.load_model('chat_model')

        # load tokenizer object
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        # load label encoder object
        with open('label_encoder.pickle', 'rb') as enc:
            lbl_encoder = pickle.load(enc)

        # parameters
        max_len = 20
        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])
        f=0
        for i in data['intents']:
            if i['tag'] == tag:
                f=1
                res=np.random.choice(i['response'])
                break
        return render_template("index.html",data=res)
    else:
        return render_template("index.html")

if __name__== '__main__':
    app.run(debug=True)