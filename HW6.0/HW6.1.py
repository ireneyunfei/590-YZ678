import numpy as np
from keras.datasets import mnist,fashion_mnist
import pandas as pd

## load in dataset
(X,Y),(test_images1,test_labels1) = mnist.load_data()
(X2,Y2),(test_images2,test_labels2) = fashion_mnist.load_data()

## sample 500 fashion mnist images as "anomalies" and insert them into the test dataset
randomize = np.arange(500)
np.random.shuffle(randomize)
test_images2 = test_images2[randomize]
test_labels2 = test_labels2[randomize]

test_labels2 = test_labels2+10
test_images= np.concatenate((test_images1,test_images2))
test_labels= np.concatenate((test_labels1,test_labels2))
print("train images shape:", X.shape)
print("test images shape (contains mnist and fashion mnist):",test_images.shape)

## shuffle fashion mnist and mnist in the test set
randomize = np.arange(len(test_images))
np.random.shuffle(randomize)
test_images = test_images[randomize]
test_labels = test_labels[randomize]


## reshape
X = X/np.max(X)
X = X.reshape(60000,28*28)
test_images = test_images/np.max(test_images)
test_images = test_images.reshape(test_images.shape[0],28*28)

n_bottleneck = 3


## build model
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
import matplotlib.pyplot as plt
from keras import callbacks

model = models.Sequential()
model.add(layers.Dense(n_bottleneck, activation='linear', input_shape=(28 * 28,)))
model.add(layers.Dense(28*28,  activation='linear'))
model.summary()

model.compile(optimizer = 'rmsprop',loss = 'mean_squared_error')

# ------------------------
# WRITE TO LOG FILES
# ------------------------
import sys
sys.stdout = open('HW6.1_log.txt', 'w')

print("#====================")
print('Training Process')
print("#====================")
history = model.fit(X,X,epochs=10,batch_size = 1000,validation_split =0.2)

model.save('HW6.1_model.h5')


print("#====================")
print('Threshold')
print("#====================")
threshold = 2.5*model.evaluate(X,X,batch_size= X.shape[0])
print("error (mse) threshold:",threshold)

X1 = model.predict(test_images)




import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('HW6.1_history.png')
plt.show()


mse = np.mean(np.power(test_images - X1, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': test_labels})

error_df['anomaly_pred'] = error_df['Reconstruction_error']>=threshold

error_df['anomaly_pred'] =error_df.apply(lambda row: 'mnist' if row.Reconstruction_error<=threshold else 'fashion', axis=1)
error_df['source'] = error_df.apply(lambda row: 'mnist' if row.True_class<=9 else 'fashion', axis=1)


print("#====================")
print('Test Confusion Matrix & Accuracy')
print("#====================")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(error_df['source'], error_df['anomaly_pred'])
print("confusion matrix:",cm)
print("test accuracy:",(cm[0,0]+cm[1,1])/test_images.shape[0])

print("#====================")
print('Fraction anomalies detected')
print("#====================")
print(cm[0,0]/(cm[0,0]+cm[0,1]))


## example
df1 = error_df[(error_df['anomaly_pred'] =='mnist') & (error_df['source'] =='fashion')]
df2 = error_df[(error_df['anomaly_pred'] =='fashion') & (error_df['source'] =='fashion')]
df3 = error_df[(error_df['anomaly_pred'] =='mnist') & (error_df['source'] =='mnist')]


X = test_images.reshape(test_images.shape[0],28,28)
X1 = X1.reshape(X1.shape[0],28,28)
#COMPARE ORIGINAL
f, ax = plt.subplots(2,3)
I1=df3.index[0]; I2=df2.index[0];I3 = df1.index[0]
ax[0,0].imshow(X[I1])
ax[1,0].imshow(X1[I1])
ax[0,1].imshow(X[I2])
ax[1,1].imshow(X1[I2])
ax[0,2].imshow(X[I3])
ax[1,2].imshow(X1[I3])
plt.savefig('HW6.1_original&reconstructed.png')
plt.show()



sys.stdout.close()