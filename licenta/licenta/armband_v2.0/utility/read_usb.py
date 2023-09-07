from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QEventLoop
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QTextStream
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import QMutex


class ReadPort(QThread):
    readyWindowsSignal = pyqtSignal(list)
    errorSignal = pyqtSignal()
    
    def __init__(self, data, windows, device, numberOfChannels):
        QThread.__init__(self)                      # Initializare thread
        self.moveToThread(self)                     # Mut toate metodele pe thread
        self.quitFlag = False
        
        self.mutex = QMutex()
        self.mutex.lock()
        self.device = device                        # Referinta catre device
        self.numberOfChannels = numberOfChannels    # Referinta catre numarul de canale folosite
        self.data = data                            # Referinta catre datele obtinute de la device
        self.mutex.unlock()
        
        self.buffer = ""                            # Buffer-ul care contine sirul de caractere primit de la device
        self.rawDataPacket = ""                     # Buffer-ul care contine sirul de caractere corescunzator unui singur pachet
        self.deltaTime = 0.002                      # Distanta in timp intre doua pachete
        
        self.windows = windows
        for _ in range(self.numberOfChannels):
            self.windows.append([])
        
        self.data['x'] = [0] * 1000
        self.data['y'] = [[0] * 1000 for _ in range(self.numberOfChannels)]
        
        self.readTimer = QTimer()                           # Crearea unui timer care are ca scop sa colecteze datele
        self.readTimer.moveToThread(self)                   # Mutarea timer-ului in thread-ul curent
        self.readTimer.timeout.connect(self.collectData)    # Apelarea functiei de colectare si prelucrare atunci cand se termina numaratoarea
        
    def run(self):      # RUN
        self.readTimer.start(100)   # Pornirea si setarea duratei timer-ului
        loop = QEventLoop()         # Crearea unui eveniment de tip loop care mentine thread-ul deschis
        loop.exec_()                # Loop
    
    def collectData(self):  # Functia care colecteaza datele
        if self.quitFlag == True:
            self.readTimer.stop()
            print("USB - transmission completed")
            self.quit()
            return
        
        self.mutex.lock()
        self.buffer = self.device.port.readAll()            # Incarcarea datelor de la device ca bytes
        self.mutex.unlock()
        
        if len(self.buffer) == 0:
            print("USB - error")
            self.errorSignal.emit()
            return
        
        self.buffer = QTextStream(self.buffer).readAll()    # Transformarea datelor in text
        
        for i in range(len(self.buffer)):               # Parcurgerea fiecarui caracter din buffer
            if self.buffer[i] != '\n':                  # Daca este diferit de caracterul de separare
                self.rawDataPacket += self.buffer[i]    # Se formeaza pachetul
            else:                                       # Altfel
                self.processPacket()                    # Se apeleaza functia care proceseaza pachetul
                self.rawDataPacket = ""                 # Stergerea pachetului
    
    def processPacket(self):    # Functia care proceseaza un pachet
        self.mutex.lock()
        time = self.data['x'][-1] + self.deltaTime # Determinarea valorii de timp pentru noul punct
        self.data['x'].append(time)                # Punerea valori in lista pentru noul punct
        self.data['x'].pop(0)                      # Eliminarea primului element
        self.mutex.unlock()
        
        for channel in range(self.numberOfChannels):  # Pentru fiecare canal
            try:
                value = int(self.rawDataPacket[2*channel : 2*channel+2], 16) - 128    # Valorea punctului
                
                self.mutex.lock()
                self.data['y'][channel].append(value)     # Punerea valori in lista canalului respectiv
                self.data['y'][channel].pop(0)            # Eliminarea primului element pentru a pastra lungimea listei 5000
                                
                self.windows[channel].append(value)
                self.mutex.unlock()
            except:
                pass
        
        q = True
        for channel in range(self.numberOfChannels):
            self.mutex.lock()
            q &= (len(self.windows[channel]) >= 128)
            self.mutex.unlock()
        if q:
            copyWindows = self.windows.copy()
            self.readyWindowsSignal.emit(copyWindows)
            self.mutex.lock()
            for _ in range(self.numberOfChannels):
                self.windows.pop(0)
            for _ in range(self.numberOfChannels):
                self.windows.append([])
            self.mutex.unlock()
    
    def stop(self):
        self.quitFlag = True
