from keras.datasets import mnist,cifar10,fashion_mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
from keras import layers
from keras import models
from keras import callbacks
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

import requests
requests.packages.urllib3.disable_warnings()
import ssl
import keras_tuner as kt


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


# ------------------------
# CODE PARAMETERS
# ------------------------
## specify which dataset to use
print("=====specify which dataset to use (mnist, fashion_mnist, cifar10)======")
dataset_name = 'cifar10'
print('dataset to use:', dataset_name)

## CNN, ANN
print('=====specify what model type to use (CNN, ANN)=====')
model_type ='CNN'
print('model type:', model_type)

#sample size to visualize
Nsample = 10

# Data Augmentation
data_aug = True

# Set Hyperparameters
print('=====Hyperparameters=====')
PARAMS = {'batch_size': 64,
          'n_epochs': 10,
          'activation': 'relu',
          '2D_units1':32,
          '2D_units2':64,
          'dense_units': 64,
          'kernel_size':(3,3),
          'pool_size':(2,2),
          'optimizer': 'rmsprop',
          'loss':'categorical_crossentropy',
          'metrics':['accuracy']
          }

## choose a dataset
if dataset_name == 'mnist':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
if dataset_name == 'fashion_mnist':
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
if dataset_name =='cifar10':
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# ------------------------
# RANDOM VISUALIZE IMAGES
# ------------------------

def visualize_images(X):
    for i in range(0,X.shape[0]):
        plt.imshow(X[i], cmap=plt.cm.gray);
        plt.show(block=False)
        plt.pause(0.1)

n_pop = train_images.shape[0]
x =np.random.randint(0, n_pop-1,Nsample)
X =train_images[x]

visualize_images(X)

# ------------------------
# DATA PREPROCESSING
# ------------------------
if len(train_images.shape)<4:
    train_images = train_images.reshape(train_images.shape + (1,))
    test_images = test_images.reshape(test_images.shape + (1,))


### partition data
f_train=0.8; f_val=0.2


#PARTITION DATA
rand_indices = np.random.permutation(train_images.shape[0])
CUT1=int(f_train*train_images.shape[0])
train_idx, val_idx = rand_indices[:CUT1], rand_indices[CUT1:]
print('------PARTITION INFO---------')
print("train_idx shape:",train_idx.shape)
print("val_idx shape:"  ,val_idx.shape)

val_images = train_images[val_idx]
val_labels = train_labels[val_idx]

train_images = train_images[train_idx]
train_labels = train_labels[train_idx]

## reshape the images, and make it [0,1]
train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2],  train_images.shape[3]))
train_images = train_images.astype('float32') / 255

val_images = val_images.reshape((val_images.shape[0], val_images.shape[1], val_images.shape[2],  train_images.shape[3]))
val_images = val_images.astype('float32') / 255

test_images = test_images.reshape((test_images.shape[0], test_images.shape[1], test_images.shape[2],  train_images.shape[3]))
test_images = test_images.astype('float32') / 255

output_size = len(np.unique(train_labels))
input_shape = (train_images.shape[1], train_images.shape[2], train_images.shape[3])

train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)

## for debugging
# train_images = train_images[0:2000]
# val_images = val_images[0:400]
# test_images = test_images[0:100]
#
# train_labels = train_labels[0:2000]
# val_labels = val_labels[0:400]
# test_labels = test_labels[0:100]


# ------------------------
# Modeling
# ------------------------
print('=====Data Augmentation and Model Building =====')

## data augmentations
data_augmentation = models.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])
resize_and_rescale = models.Sequential([
  layers.Resizing(train_images.shape[1],train_images.shape[2]),
  layers.Rescaling(1./255)
])
print('=====Hyperparameter Tuning =====')

## save model into a function for further parameter tuning process
def model_builder(hp):
    hp_units = hp.Int('units', min_value=32, max_value=128, step=32)
    if model_type =='CNN':
        if data_aug:
            model = models.Sequential([resize_and_rescale,data_augmentation])
        else:
            model = models.Sequential()
        # model.add(resize_and_rescale)
        # model.add(data_augmentation)
        model.add(layers.Conv2D(PARAMS['2D_units1'], PARAMS['kernel_size'], activation=PARAMS['activation'], input_shape=input_shape))
        model.add(layers.MaxPooling2D(PARAMS['pool_size']))
        model.add(layers.Conv2D(PARAMS['2D_units2'], PARAMS['kernel_size'], activation=PARAMS['activation']))
        model.add(layers.MaxPooling2D(PARAMS['pool_size']))
        model.add(layers.Conv2D(PARAMS['2D_units2'], PARAMS['kernel_size'], activation=PARAMS['activation']))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=hp_units, activation=PARAMS['activation']))
        model.add(layers.Dense(output_size, activation='softmax'))

    if model_type =='ANN':
        if data_aug:
            model = models.Sequential([resize_and_rescale,data_augmentation])
        else:
            model = models.Sequential()
        model.add(layers.Flatten())
        model.add(layers.Dense(units=hp_units, activation=PARAMS['activation']))
        model.add(layers.Dense(output_size, activation='softmax'))


    model.compile(optimizer=PARAMS['optimizer'],
              loss=PARAMS['loss'],
              metrics=PARAMS['metrics'])
    return model


print("=====hyper-parameter tunning=====")
tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=20)
stop_early = callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(train_images, train_labels, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} .
""")

model = tuner.hypermodel.build(best_hps)
history = model.fit(train_images, train_labels, epochs=20, validation_split=0.2)

# val_acc_per_epoch = history.history['val_accuracy']
# best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
# print('Best epoch: %d' % (best_epoch,))

# history = model.fit(train_images, train_labels, epochs=PARAMS['n_epochs'],
#                     batch_size=PARAMS['batch_size'],
#                     validation_data=(val_images,val_labels))

model.summary()

# ------------------------
# Visualizing Train and Val History
# ------------------------


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
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



# ------------------------
# SAVE AND LOAD
# ------------------------

print('=====saving and loading=====')
## save and load
filename = dataset_name
model.save(filename+'.h5')
model1 = models.load_model(filename+'.h5')

# ------------------------
# PLOT ACTIVATIONS
# ------------------------
print('=====visualize activations=====')

## choose an image
img_tensor = train_images[10]
img_tensor = np.expand_dims(img_tensor,axis = 0)

layer_outputs = [layer.output for layer in model.layers[2:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)

# visualize the first layer activation
first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, 7], cmap='viridis')
plt.show()


layer_names = []
for layer in model.layers[0:7]:
    layer_names.append(layer.name)
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations[0:5]):
    print(layer_name)
    print(layer_activation.shape)
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :,col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()