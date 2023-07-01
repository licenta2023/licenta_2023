import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


X_train = np.load('X_train_time.npy')
X_val = np.load('X_val_time.npy')
X_test = np.load('X_test_time.npy')

Y_train = np.load('Y_train_time.npy')
Y_val = np.load('Y_val_time.npy')
Y_test = np.load('Y_test_time.npy')

'''
X_train = np.load('X_train_fft.npy')
X_val = np.load('X_val_fft.npy')
X_test = np.load('X_test_fft.npy')

Y_train = np.load('Y_train_fft.npy')
Y_val = np.load('Y_val_fft.npy')
Y_test = np.load('Y_test_fft.npy')
'''
Y_train = tf.keras.utils.to_categorical(Y_train, 15)
Y_val = tf.keras.utils.to_categorical(Y_val, 15)
Y_test = tf.keras.utils.to_categorical(Y_test, 15)

# functia care defineste modelul retelei neurale
def get_model_time():
    input_layer = tf.keras.layers.Input((X_train.shape[1],))  #vector unidimensional de 3

    hidden_layer = tf.keras.layers.Dense(256, activation = 'swish')(input_layer) # repeated assignments increaste the number of layers inside network
    hidden_layer = tf.keras.layers.Dropout(0.1)(hidden_layer)
    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
    
    hidden_layer = tf.keras.layers.Dense(128, activation = 'swish')(hidden_layer)
    hidden_layer = tf.keras.layers.Dropout(0.1)(hidden_layer)
    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
    
    hidden_layer = tf.keras.layers.Dense(64, activation = 'swish')(hidden_layer)
    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
    
    hidden_layer = tf.keras.layers.Dense(256, activation = 'swish')(hidden_layer)
    #hidden_layer = tf.keras.layers.Dropout(0.1)(hidden_layer)
    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
    
    hidden_layer = tf.keras.layers.Dense(128, activation = 'swish')(hidden_layer)
    #hidden_layer = tf.keras.layers.Dropout(0.1)(hidden_layer)
    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
    
    hidden_layer = tf.keras.layers.Dense(64, activation = 'swish')(hidden_layer)
    
    output_layer = tf.keras.layers.Dense(15, activation = 'softmax')(hidden_layer)

    model = tf.keras.models.Model(inputs = [input_layer], outputs = [output_layer])
    return model

def get_model_fft():
    input_layer = tf.keras.layers.Input((X_train.shape[1],))  #vector unidimensional de 3

    hidden_layer = tf.keras.layers.Dense(512, activation = 'swish')(input_layer) # repeated assignments increaste the number of layers inside network
    hidden_layer = tf.keras.layers.Dropout(0.2)(hidden_layer)
    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
    
    hidden_layer = tf.keras.layers.Dense(256, activation = 'swish')(hidden_layer)
    hidden_layer = tf.keras.layers.Dropout(0.1)(hidden_layer)
    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
    
    hidden_layer = tf.keras.layers.Dense(128, activation = 'swish')(hidden_layer)
    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
    
    hidden_layer = tf.keras.layers.Dense(512, activation = 'swish')(hidden_layer)
    hidden_layer = tf.keras.layers.Dropout(0.2)(hidden_layer)
    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
    
    hidden_layer = tf.keras.layers.Dense(256, activation = 'swish')(hidden_layer)
    hidden_layer = tf.keras.layers.Dropout(0.1)(hidden_layer)
    hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
    
    hidden_layer = tf.keras.layers.Dense(128, activation = 'swish')(hidden_layer)
    
    hidden_layer = tf.keras.layers.Dense(64, activation = 'swish')(hidden_layer)
    
    output_layer = tf.keras.layers.Dense(15, activation = 'softmax')(hidden_layer)

    model = tf.keras.models.Model(inputs = [input_layer], outputs = [output_layer])
    return model

model = get_model_time()

device = '/gpu:0' if tf.test.is_gpu_available() else '/cpu:0'

# setari de baza pentru antrenarea retelei neurale
with tf.device(device):
    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])


checkpoint = tf.keras.callbacks.ModelCheckpoint('model_time.hdf5', monitor = 'val_loss', save_best_only = True, save_weights_only = False, verbose = 1)
#checkpoint = tf.keras.callbacks.ModelCheckpoint('model_fft.hdf5', monitor = 'val_loss', save_best_only = True, save_weights_only = False, verbose = 1)

#antrenarea retelei 

with tf.device(device):
    history = model.fit(X_train, Y_train, batch_size = 128, epochs = 80, validation_data = (X_val, Y_val), callbacks = [checkpoint])

predictions = model.predict(X_test)
model_output = np.argmax(predictions, axis=1)
true_output = np.argmax(Y_test, axis=1)

cm = confusion_matrix(true_output, model_output)

print(cm)

#evaluate the model
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy: ', test_acc)


fig, ax1 = plt.subplots()

ax1.plot(history.history['loss'], color='red', label='loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss', color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.plot(history.history['accuracy'], color='blue', label='accuracy')
ax2.set_ylabel('accuracy', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

ax1.set_title('Loss and Accuracy Training')

plt.show()


fig, ax1 = plt.subplots()

ax1.plot(history.history['val_loss'], color='red', label='loss')
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss', color='red')
ax1.tick_params(axis='y', labelcolor='red')

ax2 = ax1.twinx()
ax2.plot(history.history['val_accuracy'], color='blue', label='accuracy')
ax2.set_ylabel('accuracy', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

ax1.set_title('Loss and Accuracy Validation')

plt.show()
