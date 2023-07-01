import os
import numpy as np

class DataLoader():
    def __init__(self):
        self.folderPath = './Dataset'
        self.fileList = os.listdir(self.folderPath)
        self.npy_files = [file for file in self.fileList if file.endswith('.npy')]
        self.npy_files = sorted(self.npy_files)
        
        self.nr_channels = 8
        self.nr_features = 64
        self.nr_gestures = 15
        self.nr_users = 53
        self.nr_windows = 20
        #self.nr_permutations = 8
        self.blockSize = 128
        
        self.data = np.empty((0, self.nr_channels*self.nr_features+1))
    
    def load(self):
        for file in self.npy_files:
            print(file)
            file_path = os.path.join(self.folderPath, file)
            file_content = np.load(file_path)
            
            gestureNumber = int(file[3:5])
            
            windowedSignal = np.zeros((self.nr_channels, self.blockSize, self.nr_windows))
            for channel in range(self.nr_channels):
                originalSignal = np.array([float(k-128)/128 for k in file_content[channel]])
                originalSignal = originalSignal[:self.blockSize*self.nr_windows]
                
                windowedSignal[channel] = originalSignal.reshape((self.nr_windows, self.blockSize)).T
            
            for window in range(self.nr_windows):
                newRow = np.empty(0)
                
                newRow = np.append(newRow, gestureNumber)
                for channel in range(self.nr_channels):
                    features = np.abs(np.fft.fft(windowedSignal[channel,:,window])) / (self.blockSize / 2)
                    features = features[:int(self.blockSize / 2)]
                    newRow = np.append(newRow, features)
                self.data = np.vstack([self.data, newRow])
                '''
                for permutation in range(self.nr_permutations):
                    newRow = np.append(newRow[permutation*self.nr_features:], newRow[:permutation*self.nr_features])
                    auxNewRow = np.append(gestureNumber, newRow)
                    self.data = np.vstack([self.data, auxNewRow])
                '''
        return self.data

    def data_partitioning_window_level(self, train_percentage, validation_percentage, test_percentage):
        len_train = int(self.nr_windows * train_percentage / 100)
        len_validation = int(self.nr_windows * validation_percentage / 100)
        #len_test = int(self.nr_windows * test_percentage / 100)
        
        train = []
        validation = []
        test = []
        
        for gest in range(self.nr_users*2*self.nr_gestures):
            start = gest * self.nr_windows
            stop = start + self.nr_windows
            aux = self.data[start:stop].copy()
            aux = aux[np.random.permutation(aux.shape[0]), :]
            
            train.extend(list(aux[:len_train]))
            validation.extend(list(aux[len_train: len_train+len_validation]))
            test.extend(list(aux[len_train+len_validation:]))
        
        train = np.array(train)
        validation = np.array(validation)
        test = np.array(test)
        
        train = train[np.random.permutation(train.shape[0]), :]
        validation = validation[np.random.permutation(validation.shape[0]), :]
        test = test[np.random.permutation(test.shape[0]), :]
        
        return train, validation, test


dl = DataLoader()

data = dl.load()
train, val, test = dl.data_partitioning_window_level(70, 20, 10)

X_train = train[:, 1:]
X_val = val[:, 1:]
X_test = test[:, 1:]

Y_train = np.array([int(x) for x in train[:, 0]])
Y_val = np.array([int(x) for x in val[:, 0]])
Y_test = np.array([int(x) for x in test[:, 0]])

print(X_train.shape, Y_train.shape)
print(X_val.shape, Y_val.shape)
print(X_test.shape, Y_test.shape)
print(np.min(X_train), np.max(X_train), np.mean(X_train))
print(np.min(X_val), np.max(X_val), np.mean(X_val))
print(np.min(X_test), np.max(X_test), np.mean(X_test))

np.save('X_train_fft', X_train)
np.save('X_val_fft', X_val)
np.save('X_test_fft', X_test)

np.save('Y_train_fft', Y_train)
np.save('Y_val_fft', Y_val)
np.save('Y_test_fft', Y_test)
