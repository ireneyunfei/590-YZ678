from keras.datasets import imdb
import matplotlib
import matplotlib.pyplot as plt
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


word_index = imdb.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(
    [reverse_word_index.get(i - 3, '?') for i in train_data[0]])


import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


LR = 0.001
l1 = 1e-4
l2 = 1e-4

unit1 = 16
unit2 = 32

print("=============Training=========")
from keras import models
from keras import layers
from keras import regularizers
model = models.Sequential()
model.add(layers.Dense(unit1, activation='tanh', input_shape=(10000,),
          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
model.add(layers.Dense(unit2, activation='relu',
          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
model.add(layers.Dense(1, activation='sigmoid',
          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
from tensorflow.keras import optimizers
# model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])


from keras import losses
from keras import metrics
model.compile(optimizer=optimizers.RMSprop(learning_rate=LR),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

print("=============Loss and Acc during Training=========")

import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc =history_dict['acc']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()




print("========Fit the model with the best epoch=========")


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,),
          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
model.add(layers.Dense(16, activation='relu',
          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
model.add(layers.Dense(1, activation='sigmoid',
          kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=512)
results = model.evaluate(x_test, y_test)

print("training set loss and acc:" ,[history_dict['loss'][-1],history_dict['acc'][-1]])
print("validation set loss and acc:" ,[history_dict['val_loss'][-1],history_dict['val_acc'][-1]])
print("test results:", results)




