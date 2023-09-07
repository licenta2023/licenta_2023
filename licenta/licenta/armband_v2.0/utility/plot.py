from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QEventLoop
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QMutex

import numpy as np

class Plot(QThread):
    def __init__(self, processedData, autocorrelationData, numberOfChannels, widgetList, plotWidget_autocorrelation, settings):
        QThread.__init__(self)                      # Initializare thread
        self.mutex = QMutex()
        
        self.processedData = processedData          # Referinta catre datele obtinute de la device
        self.autocorrelationData = autocorrelationData
        self.numberOfChannels = numberOfChannels    # Referinta catre numarul de canale folosite
        self.widgetList = widgetList                # Referinta catre widget-urile unde se face desenarea
        self.plotWidget_autocorrelation = plotWidget_autocorrelation
        self.settings = settings                    # Referinta catre setarile canalelor
        
        self.plotFlag = 0
        x = np.linspace(1, 128, 128)
        y = np.linspace(1, 128, 128)
        self.k1, self.k2 = np.meshgrid(x, y)
        
        self.timer = QTimer()                   # Crearea unui timer care are ca scop sa ploteze
        self.timer.moveToThread(self)           # Mutarea timer-ului in thread-ul curent
        self.timer.timeout.connect(self.plot)   # Apelarea functiei de plotare atunci cand se termina numaratoarea
    
    def run(self):      # RUN
        self.timer.start(100)   # Pornirea si setarea duratei timer-ului
        loop = QEventLoop()     # Crearea unui eveniment de tip loop care mentine thread-ul deschis
        loop.exec_()            # Loop
    
    def plot(self):     # Functia care ploteaza  
        for channel in range(self.numberOfChannels):      # Pentru fiecare canal
            if self.settings[channel] == 'autocorrelation':
                if self.autocorrelationData[channel] is not None:
                    if self.plotFlag == 20:
                        self.plotFlag = 0
                        self.mutex.lock()
                        self.plotWidget_autocorrelation[channel].ax.cla()
                        self.plotWidget_autocorrelation[channel].ax.plot_surface(self.k1, self.k2, self.autocorrelationData[channel], cmap='viridis')
                        self.plotWidget_autocorrelation[channel].canvas.draw()
                        self.mutex.unlock()
                    else:
                        self.plotFlag += 1
            else:
                self.mutex.lock()
                if self.processedData['x'][channel] is not None:
                    if len(self.processedData['x'][channel]) < len(self.processedData['y'][channel]):
                        self.widgetList[channel].plotcurve.setData(self.processedData['x'][channel],
                                                                   self.processedData['y'][channel][0:len(self.processedData['x'][channel])], 
                                                                   pen=self.widgetList[channel].pen, 
                                                                   clickable=True)           # Se updateaza graficul
                    else:
                        self.widgetList[channel].plotcurve.setData(self.processedData['x'][channel][0:len(self.processedData['y'][channel])],
                                                                   self.processedData['y'][channel], 
                                                                   pen=self.widgetList[channel].pen, 
                                                                   clickable=True)           # Se updateaza graficul
                self.mutex.unlock()
    
    def close(self):
        print("Close Plot")
        self.quit()
