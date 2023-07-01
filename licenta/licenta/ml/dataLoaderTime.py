import os
import numpy as np

class DataLoader():
    def __init__(self):
        self.folderPath = './Dataset'
        self.fileList = os.listdir(self.folderPath)
        self.npy_files = [file for file in self.fileList if file.endswith('.npy')]
        self.npy_files = sorted(self.npy_files)
        
        self.nr_channels = 8
        self.nr_features = 8
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
                originalSignal = np.array([float(k-128) for k in file_content[channel]])
                originalSignal = originalSignal[:self.blockSize*self.nr_windows]
                
                windowedSignal[channel] = originalSignal.reshape((self.nr_windows, self.blockSize)).T
            
            for window in range(self.nr_windows):
                newRow = np.empty(0)
                
                newRow = np.append(newRow, gestureNumber)
                for channel in range(self.nr_channels):
                    features = self.features_calc(windowedSignal[channel,:,window])
                    newRow = np.append(newRow, features)
                self.data = np.vstack([self.data, newRow])
                '''
                for permutation in range(self.nr_permutations):
                    newRow = np.append(newRow[permutation*self.nr_features:], newRow[:permutation*self.nr_features])
                    auxNewRow = np.append(gestureNumber, newRow)
                    self.data = np.vstack([self.data, auxNewRow])
                '''
        self.norm()
        return self.data
    
    def norm(self):
        maxim = [0] * self.nr_features
        for feature in range(self.nr_features):
            aux = np.empty(0)
            for channel in range(self.nr_channels):
                index = (channel * 8) + feature + 1
                aux = np.append(aux, self.data[:, index])
            maxim[feature] = np.max(np.abs(aux))
            for channel in range(self.nr_channels):
                index = (channel * 8) + feature + 1
                self.data[:, index] = self.data[:, index] / float(maxim[feature])
        np.save('normalization', np.array(maxim))

    def features_calc(self, w):
        MAV = self.meanAbsoluteValue(w)
        SSC = self.slopeSignChanges(w)
        ZCR = self.zeroCrossRate(w)
        WL = self.waveformLength(w)
        Skewness = self.skewness(w)
        RMS = self.rootMeanSquare(w)
        Hjorth = self.hjorthActivity(w)
        iEMG = self.integratedEMG(w)
        return np.array([MAV, SSC, ZCR, WL, Skewness, RMS, Hjorth, iEMG])
    
    def meanAbsoluteValue(self, signal):
        return np.mean(np.abs(signal))

    def slopeSignChanges(self, signal):
        diff = signal[1:] - signal[:-1]
        return np.sum((diff[:-1] * diff[1:]) < 0)

    def zeroCrossRate(self, signal):
        return np.sum((signal[:-1] * signal[1:]) < 0)

    def waveformLength(self, signal): # lungimea formei semnalului
        return np.sum(np.abs(signal[1:] - signal[:-1]))

    def skewness(self, signal):
        std = np.std(signal)
        mean = np.mean(signal)
        cen = signal - mean
        return np.mean((cen / (std + 1e-3)) ** 3)

    def rootMeanSquare(self, signal):
        return np.sqrt(np.mean(signal ** 2))
    
    def hjorthActivity(self, signal): # varianta semnalului
        mean = np.mean(signal)
        cen = signal - mean
        return np.mean(cen ** 2)

    def integratedEMG(self, signal):
        return np.sum(np.abs(signal))

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

np.save('X_train_time', X_train)
np.save('X_val_time', X_val)
np.save('X_test_time', X_test)

np.save('Y_train_time', Y_train)
np.save('Y_val_time', Y_val)
np.save('Y_test_time', Y_test)
