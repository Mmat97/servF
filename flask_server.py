from flask import Flask, request
import functions
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET', 'POST'])
def welcome():

    if request.method == 'POST':
        data = request.get_json()
        query = data['type']

        if query == 'Text':
            words = data['input']
            emotion = functions.text_classifier(words)
            return emotion
        if query == 'Audio':
            path = 'audio_files/' + data['input']
            words = functions.transcribe(path)
            emotion = functions.audio_classifier(path)
            print(words, emotion)
            response = words + '@' + emotion
            return response
        if query == 'Chatbot':

            words = data['input']
            emotion = data['emotion']


            print("sdfsdfsffsdfsf")


            response = functions.chat(emotion, words)




            return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)