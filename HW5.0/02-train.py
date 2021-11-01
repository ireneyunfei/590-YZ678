import pandas as pd
import os
import keras_tuner as kt
from datetime import datetime
# ------------------------
# CODE PARAMETERS
# ------------------------
model_type = 'CNN'

print('=====Hyperparameters=====')
PARAMS = {'batch_size': 64,
          'n_epochs': 20,
          'n_cov1d_layer':1,
          'activation': 'relu',
          '1D_units1':32,
          '1D_units2':64,
          'dense_units': 32,
          'kernel_size':3,
          'pool_size':2,
          'return_sequence':False,
          'l2':10e-3,
          'rnn_unit':32,
          'recurrent_dropout': 0.2,
          'optimizer': 'rmsprop',
          'loss' : 'categorical_crossentropy',
          'metrics':['accuracy']
          }


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

## Tokenization

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

# ------------------------
# pre-trained embedding
# ------------------------

glove_dir = '/Users/irene/Downloads/glove.6B'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


# ------------------------
# build the model
# ------------------------
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras import models, layers
from keras import optimizers
from keras import regularizers
from keras.layers import SimpleRNN
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import roc_curve, auc, roc_auc_score

def build_model():

    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))

    # FIRST LAYER
    if (model_type == "CNN"):
        model.add(layers.Conv1D(PARAMS['1D_units1'],
                                kernel_initializer="ones",
                                bias_initializer="zeros",
                                kernel_size=PARAMS['kernel_size'],
                                activation=PARAMS['activation'],
                                input_shape =(max_words,embedding_dim)))
        model.add(layers.MaxPooling1D(PARAMS['pool_size']))

    # ADD CONV LAYERS AS IF NEEDED:
    if (model_type == "CNN"):
        for i in range(1, PARAMS['n_cov1d_layer'] - 1):
            model.add(layers.Conv1D(PARAMS['1D_units2'],
                                    kernel_size=PARAMS['kernel_size'],
                                    activation=PARAMS['activation']))
            model.add(layers.MaxPooling1D(PARAMS['pool_size']))
        model.add(layers.Conv1D(PARAMS['1D_units2'],
                                kernel_size=PARAMS['kernel_size'],
                                activation=PARAMS['activation']))
        model.add(layers.Flatten())

    if(model_type=='RNN'):
        model.add(SimpleRNN(PARAMS['rnn_unit'],
                            recurrent_dropout=PARAMS['recurrent_dropout'],
                            kernel_regularizer=regularizers.l2(l2 = PARAMS['l2']),
                            return_sequences=PARAMS['return_sequence'],
                            activation = PARAMS['activation']))


    model.add(Dense(16, activation='relu',
                    kernel_regularizer=regularizers.l2(l2 = PARAMS['l2'])))
    model.add(Dense(3, activation='softmax'))

    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    model.summary()



    model.compile(optimizer=PARAMS['optimizer'],
              loss=PARAMS['loss'],
              metrics=PARAMS['metrics'])
    return model

# ------------------------
# WRITE TO LOG FILES
# ------------------------
import sys
if model_type =='CNN':
    sys.stdout = open('cnn_log.txt', 'w')
if model_type =='RNN':
    sys.stdout = open('rnn_log.txt', 'w')

# from tensorflow import keras
# logdir = "./logs"
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

## fit the model
model = build_model()
history = model.fit(x_train, y_train,epochs=PARAMS['n_epochs'], batch_size=PARAMS['batch_size'],validation_data=(x_val, y_val))
  #  callbacks=[tensorboard_callback])

## save the model

if(model_type=='CNN'):
    model.save('CNN.h5')
if(model_type=='RNN'):
    model.save('RNN.h5')


## plotting
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
if(model_type =='CNN'):
    plt.savefig('cnn_acc.png')
if (model_type == 'RNN'):
    plt.savefig('rnn_acc.png')

plt.figure()


plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
if(model_type =='CNN'):
    plt.savefig('cnn_loss.png')
if (model_type == 'RNN'):
    plt.savefig('rnn_loss.png')
plt.show()


# ------------------------
# TRACK ROC AND AUC
# ------------------------

ypred = model.predict(x_train)


print('ROC AUC score:',roc_auc_score(y_train, ypred, average='macro'))

ypred1 = ypred.argmax(axis=-1)
ytrain1 = y_train.argmax(axis=-1)


## ROC and AUC for each book (each category)
target= ['biology', 'monte_cristo', 'psychology']


lb = MultiLabelBinarizer()
y_test =[str(int) for int in ytrain1]
y_test = np.array(y_test)
lb.fit(y_test)
y_test = lb.transform(y_test)
y_pred =[str(int) for int in ypred1]
y_pred = np.array(y_pred)
y_pred = lb.transform(y_pred)

print("AUC for each book")
for (idx, c_label) in enumerate(target):
    fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
    #c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    print((c_label, auc(fpr, tpr)))
    #c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')



sys.stdout.close()