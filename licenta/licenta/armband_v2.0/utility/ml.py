from PyQt5.QtCore import QEventLoop
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import pyqtSlot

import tensorflow as tf
import numpy as np
import pickle

class ML(QThread):
    doneSignal = pyqtSignal()
    doneClassification = pyqtSignal(int)
    
    def __init__(self, numberOfChannels):
        QThread.__init__(self)  # Initializare thread
        self.moveToThread(self) # Mut toate metodele pe thread
        
        self.numberOfChannels = numberOfChannels
        
        self.majorityRule = []
        
        self.model_time = {}
        self.normalization = {}
        self.model_fft = {}
        
        with open('./ml/models/my_nn_time_5.pickle', 'rb') as file:
            my_nn_time_5 = pickle.load(file)
        
        with open('./ml/models/my_nn_fft_5.pickle', 'rb') as file:
            my_nn_fft_5 = pickle.load(file)
        
        with open('./ml/models/my_nn_time_15.pickle', 'rb') as file:
            my_nn_time_15 = pickle.load(file)
        
        self.model_fft['NN - 5 gesturi - caracteristici in frecventa - TensorFlow'] = tf.keras.models.load_model('./ml/models/model_fft_5.hdf5')
        self.model_fft['NN - 5 gesturi - caracteristici in frecventa - Versiunea mea'] = my_nn_fft_5
        self.model_fft['NN - 15 gesturi - caracteristici in frecventa - TensorFlow'] = tf.keras.models.load_model('./ml/models/model_fft_15.hdf5')
        
        self.model_time['NN - 5 gesturi - caracteristici in timp - TensorFlow'] = tf.keras.models.load_model('./ml/models/model_time_5.hdf5') # primul model de retea neurala
        self.model_time['NN - 5 gesturi - caracteristici in timp - Versiunea mea'] = my_nn_time_5
        self.model_time['NN - 15 gesturi - caracteristici in timp - TensorFlow'] = tf.keras.models.load_model('./ml/models/model_time_15.hdf5')
        self.model_time['NN - 15 gesturi - caracteristici in timp - Versiunea mea'] = my_nn_time_15
        
        self.normalization['NN - 5 gesturi - caracteristici in timp - TensorFlow'] = np.load('./ml/models/normalization_5.npy')  # valori necesare pentru normalizarea datelor
        self.normalization['NN - 5 gesturi - caracteristici in timp - Versiunea mea'] = np.load('./ml/models/normalization_5.npy')
        self.normalization['NN - 15 gesturi - caracteristici in timp - TensorFlow'] = np.load('./ml/models/normalization_15.npy')
        self.normalization['NN - 15 gesturi - caracteristici in timp - Versiunea mea'] = np.load('./ml/models/normalization_15.npy')
    
    def run(self):      # RUN
        loop = QEventLoop()         # Crearea unui eveniment de tip loop care mentine thread-ul deschis
        loop.exec_()                # Loop
    
    def close(self):
        print("Close ML")
        self.quit()
    
    def emitSignal(self):
        count = {}
        for gesture in self.majorityRule:
            count[gesture] = count.get(gesture, 0) + 1
        
        max_count = max(count.values())
        majority_values = [value for value, freq in count.items() if freq == max_count]
        result = majority_values[-1]
        #print(result)
        
        self.majorityRule = []
        self.doneClassification.emit(result)
        
    @pyqtSlot(list)
    def classificationFFT(self, arg):
        #if QThread.currentThread() == self.thread():
        #    print("Running on a separate thread")
        #else:
        #    print("Not running on a separate thread")
        
        window = arg[0]
        model = arg[1]
        
        self.window = np.array([[float(element) for element in sublist] for sublist in window])
        
        inp = np.empty(0)
        self.window /= 128.0 # normalizare
        for channel in range(self.numberOfChannels):
            absFFT = np.abs(self.window[channel]) / (len(self.window[channel]) / 2)
            absFFT = absFFT[:int(len(absFFT)/2)]          
            inp = np.append(inp, absFFT)
        
        if model in ['NN - 15 gesturi - caracteristici in frecventa - Versiunea mea', 'NN - 5 gesturi - caracteristici in frecventa - Versiunea mea']:
            inp = np.reshape(inp, (512, 1))
        else:
            inp = np.reshape(inp, (1, 512))
        
        classification = self.model_fft[model].predict(inp)
        
        if len(self.majorityRule) == 4:
            self.emitSignal()
        
        self.majorityRule.append(int(np.argmax(classification)))
        self.doneSignal.emit()
    
    @pyqtSlot(list)
    def classificationTime(self, arg):
        window = arg[0]
        model = arg[1]
        
        self.window = np.array([np.array([float(element) for element in sublist]) for sublist in window])
                
        inp = np.empty(0)
        for channel in range(self.numberOfChannels):
            features = self.features_extraction(self.window[channel])
            
            for index in range(len(features)):
                features[index] /= self.normalization[model][index] # normalizare
            
            inp = np.append(inp, features)
        
        if model in ['NN - 15 gesturi - caracteristici in timp - Versiunea mea', 'NN - 5 gesturi - caracteristici in timp - Versiunea mea']:
            inp = np.reshape(inp, (64, 1))
        else:
            inp = np.reshape(inp, (1, 64))
        
        classification = self.model_time[model].predict(inp)
        
        if len(self.majorityRule) == 4:
            self.emitSignal()
        
        self.majorityRule.append(int(np.argmax(classification)))
        self.doneSignal.emit()
    
    def features_extraction(self, window): # functia care extrage trasaturile
        MAV = self.meanAbsoluteValue(window)
        SSC = self.slopeSignChanges(window)
        ZCR = self.zeroCrossRate(window)
        WL = self.waveformLength(window)
        Skewness = self.skewness(window)
        RMS = self.rootMeanSquare(window)
        Hjorth = self.hjorthActivity(window)
        iEMG = self.integratedEMG(window)
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
    
    def hjorthActivity(self, signal):
        mean = np.mean(signal)
        cen = signal - mean
        return np.mean(cen ** 2)

    def integratedEMG(self, signal):
        return np.sum(np.abs(signal))
