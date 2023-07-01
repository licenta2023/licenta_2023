from PyQt5.QtCore import QEventLoop
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import QMutex

import numpy as np
import scipy


class Processing(QThread):
    
    def __init__(self, data, processedData, settings, numberOfChannels):
        QThread.__init__(self)      # Initializare thread
        self.moveToThread(self)     # Mut toate metodele pe thread
        self.mutex = QMutex()
        
        self.data = data
        self.processedData = processedData
        self.settings = settings
        self.numberOfChannels = numberOfChannels
        
    def run(self):      # RUN
        loop = QEventLoop()         # Crearea unui eveniment de tip loop care mentine thread-ul deschis
        loop.exec_()                # Loop
    
    @pyqtSlot(list)
    def process(self, arg):
        #if QThread.currentThread() == self.thread():
        #    print("Running on a separate thread")
        #else:
        #    print("Not running on a separate thread")
        if self.data['x'] is None:
            return
        if self.data['y'] is None:
            return

        self.filter_coeffs_fir = np.array(arg[0])
        self.filter_coeffs_iir_a = np.array(arg[1])
        self.filter_coeffs_iir_b = np.array(arg[2])
        
        for channel in range(self.numberOfChannels):
            if self.settings[channel] is None:
                self.mutex.lock()
                self.processedData['x'][channel] = self.data['x']
                self.processedData['y'][channel] = self.data['y'][channel]
                self.mutex.unlock()
            elif self.settings[channel] == 'fftAbs':
                self.fftAbs(channel)
            elif self.settings[channel] == 'fftPhase':
                self.fftPhase(channel)
            elif self.settings[channel] == 'fir':
                self.fir(channel)
            elif self.settings[channel] == 'iir':
                self.iir(channel)
            elif self.settings[channel] == 'mean':
                self.mean(channel)
            elif self.settings[channel] == 'autocorrelation':
                self.autocorrelation(channel)
    
    def fftAbs(self, channel):
        self.mutex.lock()
        signal = np.array(self.data['y'][channel])
        samplingRate = 512
        
        FFT = np.fft.fft(signal)
        absFFT = list(np.abs(FFT) / (len(signal)/2))
        
        freq = list(np.fft.fftfreq(len(signal), 1/samplingRate))
        
        self.processedData['x'][channel] = freq
        self.processedData['y'][channel] = absFFT
        self.mutex.unlock()
    
    def fftPhase(self, channel):
        self.mutex.lock()
        signal = np.array(self.data['y'][channel])
        samplingRate = 512
        
        FFT = np.fft.fft(signal)
        phasesFFT = list(np.angle(FFT))
        
        freq = list(np.fft.fftfreq(len(signal), 1/samplingRate))
        
        self.processedData['x'][channel] = freq
        self.processedData['y'][channel] = phasesFFT
        self.mutex.unlock()
    
    def fir(self, channel):
        self.mutex.lock()
        signal = self.data['y'][channel]
        filtered_signal = scipy.signal.lfilter(self.filter_coeffs_fir, 1, signal)
        
        samplingRate = 512
        
        FFT = np.fft.fft(filtered_signal)
        absFFT = list(np.abs(FFT) / (len(filtered_signal)/2))
        
        freq = list(np.fft.fftfreq(len(filtered_signal), 1/samplingRate))
        
        self.processedData['x'][channel] = freq
        self.processedData['y'][channel] = absFFT
        
        #self.processedData['x'][channel] = self.data['x']
        #self.processedData['y'][channel] = filtered_signal
        self.mutex.unlock()
    
    def iir(self, channel):
        self.mutex.lock()
        signal = self.data['y'][channel]
        filtered_signal = scipy.signal.lfilter(self.filter_coeffs_iir_b, self.filter_coeffs_iir_a, signal)
        
        samplingRate = 512
        
        FFT = np.fft.fft(filtered_signal)
        absFFT = list(np.abs(FFT) / (len(filtered_signal)/2))
        
        freq = list(np.fft.fftfreq(len(filtered_signal), 1/samplingRate))
        
        self.processedData['x'][channel] = freq
        self.processedData['y'][channel] = absFFT
        
        #self.processedData['x'][channel] = self.data['x']
        #self.processedData['y'][channel] = filtered_signal
        self.mutex.unlock()
    
    def mean(self, channel):
        pass
    
    def autocorrelation(self, channel):
        pass
    
    def stop(self):
        self.quit()
