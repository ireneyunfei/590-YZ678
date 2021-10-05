from keras.datasets import reuters
import matplotlib.pyplot as plt
from keras import regularizers

import requests
requests.packages.urllib3.disable_warnings()
import ssl


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

len(train_data)
len(test_data)

l1 = 1e-4
l2 = 1e-4

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

# from keras.utils.np_utils import to_categorical
#
# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)

from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,),
          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
model.add(layers.Dense(64, activation='relu',
          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
model.add(layers.Dense(46, activation='softmax',
          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


history_dict = history.history
plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,),
          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
model.add(layers.Dense(64, activation='relu',
          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
model.add(layers.Dense(46, activation='softmax',
          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=10,
          batch_size=512,
          validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)

print("training set loss and acc:" ,[history_dict['loss'][-1],history_dict['accuracy'][-1]])
print("validation set loss and acc:" ,[history_dict['val_loss'][-1],history_dict['val_accuracy'][-1]])
print("test results:", results)


