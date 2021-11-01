import pandas as pd
import os

model_type = 'RNN'

# ------------------------
# LOAD IN DATA
# ------------------------
with open("texts.txt", "r") as f2:
    text = f2.read()
    texts = text.split('\n\n\t\n')

texts = texts[:3000]

a = pd.read_csv('labels.csv')
labels = a.to_numpy()

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 200
training_samples = 2000
validation_samples = 500
max_words = 10000

# ------------------------
# TOKENIZATION
# ------------------------

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]


x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]
x_test = data[training_samples + validation_samples:]
y_test = labels[training_samples + validation_samples:]

# ------------------------
# LOAD IN MODEL
# ------------------------
from keras import models, layers

if(model_type=='CNN'):
    model = models.load_model('CNN.h5')

if(model_type=='RNN'):
    model = models.load_model('RNN.h5')

# ------------------------
# WRITE THE METRIC
# ------------------------
print("model:",model_type)
print('===EVALUATION===')
print('train loss and acc:')
model.evaluate(x_train,y_train)
print('val loss and acc:')
model.evaluate(x_val,y_val)
print('test loss and acc:')
model.evaluate(x_test,y_test)






