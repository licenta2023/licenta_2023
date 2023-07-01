from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QEventLoop
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QMutex


class Plot(QThread):
    def __init__(self, processedData, numberOfChannels, widgetList):
        QThread.__init__(self)                      # Initializare thread
        self.mutex = QMutex()
        
        self.processedData = processedData          # Referinta catre datele obtinute de la device
        self.numberOfChannels = numberOfChannels    # Referinta catre numarul de canale folosite
        self.widgetList = widgetList                # Referinta catre widget-urile unde se face desenarea
        
        self.timer = QTimer()                   # Crearea unui timer care are ca scop sa ploteze
        self.timer.moveToThread(self)           # Mutarea timer-ului in thread-ul curent
        self.timer.timeout.connect(self.plot)   # Apelarea functiei de plotare atunci cand se termina numaratoarea
    
    def run(self):      # RUN
        self.timer.start(100)   # Pornirea si setarea duratei timer-ului
        loop = QEventLoop()     # Crearea unui eveniment de tip loop care mentine thread-ul deschis
        loop.exec_()            # Loop
    
    def plot(self):     # Functia care ploteaza  
        for i in range(self.numberOfChannels):      # Pentru fiecare canal
            self.mutex.lock()
            if self.processedData['x'][i] is not None:
                self.widgetList[i].plotcurve.setData(self.processedData['x'][i][0:len(self.processedData['y'][i])],
                                                     self.processedData['y'][i], 
                                                     pen=self.widgetList[i].pen, 
                                                     clickable=True)           # Se updateaza graficul
            self.mutex.unlock()
    
    def close(self):
        print("Close Plot")
        self.quit()
