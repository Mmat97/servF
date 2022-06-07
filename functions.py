import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import pandas as pd
import librosa
import pickle
from pydub import AudioSegment
import speech_recognition as sr
from tqdm import tqdm
import os
import subprocess
from parlai.scripts.interactive import Interactive
from parlai.scripts.train_model import TrainModel
import warnings
import sys


def text_classifier(input_string):
    input_string = [input_string]
    length_two = 50
    truncating_and_padding_state = 'post'

    label_matched_emotion = {0: 'surprise', 1: 'sad', 2: 'angry', 3: 'happy', 4: 'fear'}

    with open('tokenizer.json') as f:
        data = json.load(f)
        tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

    padded_inputs=pad_sequences(tokenizer.texts_to_sequences(input_string), maxlen=length_two,
                                padding=truncating_and_padding_state, truncating=truncating_and_padding_state)

    best_model = keras.models.load_model('./saved_models/best_text.h5')

    predicted = best_model.predict(np.expand_dims(padded_inputs[0], axis=0))[0]
    emotion = label_matched_emotion.get(np.argmax(predicted))

    return emotion






def audio_classifier(file_path):
    path = file_path.split('.')
    if path[1] == 'm4a':
        track = AudioSegment.from_file(file_path,  format= 'm4a')
        file_path = path[0] + '.wav'
        track.export(file_path, format='wav')








    loaded_model = pickle.load(open('saved_models/finalized_model.sav', 'rb'))


    data,sample_rate = librosa.load(file_path)
    result=np.array([])
    mfccs=np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40).T, axis=0)
    result=np.hstack((result, mfccs))
    stft=np.abs(librosa.stft(data))
    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    result=np.hstack((result, chroma))
    mel=np.mean(librosa.feature.melspectrogram(data, sr=sample_rate).T,axis=0)
    result=np.hstack((result, mel))
    result=result.reshape(1,-1)


    
    prediction=loaded_model.predict(result)
    print (prediction[0])



        


    ''' 
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=2.5, sr=44100, offset=0.5)
    sample_rate = np.array(sample_rate)


    print('00000000')
    means = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=13), axis=0)
    new_df = pd.DataFrame(data=means).T
    new_df = np.expand_dims(new_df, axis=2)


    print('111111111111')
    loaded_model = tf.keras.models.load_model('./saved_models/best_emotion.h5')
    print('22222222')
    print(loaded_model)



    le = pickle.load(open('labels.txt', 'rb'))


    print(le)

    

    print('kkkkkkkkkkk')

    prediction = loaded_model.predict(new_df, batch_size=16, verbose=1)


    
    print('mmmmmmmmmm')



    final = prediction.argmax(axis=1)
    print(final)
    print('nnnnnnnnnn')
    final = final.astype(int).flatten()
    emotion = (le.inverse_transform(final))[0].split('_')[1]
    print(emotion)

    ''' 

    return prediction[0].split('_')[0]


def transcribe(file_path):
    path = file_path.split('.')
    if path[1] == 'm4a':
        track = AudioSegment.from_file(file_path,  format= 'm4a')
        file_path = path[0] + '.wav'
        track.export(file_path, format='wav')

    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
        try:
            transcript = r.recognize_google(audio)
        except:
            print('bad')
            transcript = ''
    return transcript

def chat(emotion, text):




    models_fp = './saved_models/'


    print(text)

    if emotion.lower() not in ['happy', 'fear', 'sad', 'surprise', 'angry']:
        ss='you sound '+str(emotion.lower()).split('_')[0] + ', what do you wish to vent about?'
        return ss
    
    # run program










    start_script = 'python chat.py {0} {1}'.format(emotion, models_fp)
    print(start_script)
    p = subprocess.Popen(start_script, shell=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    print("rrrrrrrrrrr")
    # process through all the BS command line outputs
    output_terminate = '[EXIT]'
    it1_read = p.stdout.readline().decode()
    print(it1_read)
    while(output_terminate not in it1_read):
        it1_read = p.stdout.readline().decode()
        #print(it1_read)
    it1_read = p.stdout.readline().decode()
    #print(it1_read)
    print("hhhhhhhhhhh")
    # first input
    text_input = text + '\n'
    p.stdin.write(text_input.encode())
    p.stdin.flush()
    # first response
    one_line_output = p.stdout.readline().decode().replace('Enter Your Message: [TransformerGenerator]: ', '')

    response_output = one_line_output

    print(response_output)
    # exit prompt
    text_input = '[EXIT]\n'
    p.stdin.write(text_input.encode())
    p.kill()







    print("thats too bad")
    #response_output='thats too bad'
    return response_output


# test_sentence = 'i left with my bouquet of red and yellow tulips under my arm feeling slightly more optimistic than when i arrived'
# print(text_classifier(test_sentence))
#
# test_file = './Data/SAVEE/JE_sa08.wav'
# print(audio_classifier(test_file))
#
# test_file = 'SadRecording.m4a'
# print(audio_classifier(test_file))
#
# test_file = './Data/SAVEE/JE_sa08.wav'
# print(transcribe(test_file))
#
# test_file = 'SadRecording.m4a'
# print(transcribe(test_file))

# test_sentence = 'the movie was so scary'
# emotion = 'fear'
# print(chat(emotion, test_sentence))


