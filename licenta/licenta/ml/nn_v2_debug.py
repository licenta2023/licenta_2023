import numpy as np
import tensorflow as tf

X_train = np.load('X_train_fft.npy')
X_val = np.load('X_val_fft.npy')
X_test = np.load('X_test_fft.npy')

Y_train = np.load('Y_train_fft.npy')
Y_val = np.load('Y_val_fft.npy')
Y_test = np.load('Y_test_fft.npy')

print(X_train.shape, Y_train.shape)
print(np.min(X_train), np.max(X_train), np.mean(X_train))
print(np.min(X_val), np.max(X_val), np.mean(X_val))
print(np.min(Y_train), np.max(Y_train))

Y_train = np.array(tf.keras.utils.to_categorical(Y_train, 15))
Y_val = np.array(tf.keras.utils.to_categorical(Y_val, 15))
Y_test = np.array(tf.keras.utils.to_categorical(Y_test, 15))

print(Y_train.shape)

#%%
import numpy as np
import tensorflow as tf

X_train = np.load('X_train_time.npy')
X_val = np.load('X_val_time.npy')
X_test = np.load('X_test_time.npy')

Y_train = np.load('Y_train_time.npy')
Y_val = np.load('Y_val_time.npy')
Y_test = np.load('Y_test_time.npy')

print(X_train.shape, Y_train.shape)
print(np.min(X_train), np.max(X_train), np.mean(X_train))
print(np.min(X_val), np.max(X_val), np.mean(X_val))
print(np.min(Y_train), np.max(Y_train))

Y_train = np.array(tf.keras.utils.to_categorical(Y_train, 15))
Y_val = np.array(tf.keras.utils.to_categorical(Y_val, 15))
Y_test = np.array(tf.keras.utils.to_categorical(Y_test, 15))

print(Y_train.shape)

#%%
from nn_v2 import NeuralNetwork

nn = NeuralNetwork(inputSize=X_train.shape[1])

nn.addLayer(newLayerSize=64, activationFunction='relu')
nn.addLayer(newLayerSize=256, activationFunction='relu')
nn.addLayer(newLayerSize=32, activationFunction='relu')
nn.addLayer(newLayerSize=15, activationFunction='softmax')

nn.fit(X_train, Y_train, X_val, Y_val, batchSize=256, epochs=150, learningRate=0.001, decayRate=1)

nn.lossAndAccuracyPlot()

#%%
from nn_v2 import NeuralNetwork

nn = NeuralNetwork(inputSize=X_train.shape[1])

nn.addLayer(newLayerSize=512, activationFunction='relu')
nn.addLayer(newLayerSize=128, activationFunction='relu')
nn.addLayer(newLayerSize=64, activationFunction='relu')
nn.addLayer(newLayerSize=32, activationFunction='relu')
nn.addLayer(newLayerSize=15, activationFunction='softmax')

nn.fit(X_train, Y_train, X_val, Y_val, batchSize=256, epochs=50, learningRate=0.001, decayRate=1)

nn.lossAndAccuracyPlot()

#%%
import pickle

with open('my_nn_time.pickle', 'wb') as file:
    pickle.dump(nn, file)

#%%
import pickle

with open('my_nn_time.pickle', 'rb') as file:
    nn_load = pickle.load(file)

nn_load.lossAndAccuracyPlot()
