from PyQt5.QtCore import QTimer
from PyQt5.QtCore import QEventLoop
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import QMutex

import socket
import time
import numpy as np

class ReadUDP(QThread):
    readyWindowsSignal = pyqtSignal(list)
    errorSignal = pyqtSignal()
    
    def __init__(self, data, windows, numberOfChannels):
        QThread.__init__(self)                      # Initializare thread
        self.moveToThread(self)                     # Mut toate metodele pe thread
        self.quitFlag = False
        self.errorConnectionFlag = False
        
        self.counterTimeout = 0
        
        UDP_IP    =   "192.168.4.1"         # IP address of the ESP32 SoftAP
        UDP_PORT  =   1234                  # Port used for UDP transmission
        self.start_cmd =   "START".encode("utf-8")
        self.retry_cmd =   "RETRANSMIT".encode("utf-8")
        
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) 
            self.sock.settimeout(0.002)
            self.sock.connect((UDP_IP, UDP_PORT))
            #self.sock.send(self.start_cmd)
            print("UDP - connected")
        except:
            print("UDP - connection failed")
            self.errorConnectionFlag = True
        
        self.start_moment = time.time_ns()
        self.old_time = 0
        
        self.mutex = QMutex()
        self.mutex.lock()                        # Referinta catre device
        self.numberOfChannels = numberOfChannels    # Referinta catre numarul de canale folosite
        self.data = data                            # Referinta catre datele obtinute de la device
        self.mutex.unlock()
        
        self.buffer = []                           # Buffer-ul care contine sirul de numere primit de la device
        
        self.windows = windows
        
        self.data['x'] = [0] * 1024
        self.data['y'] = [[0] * 1024 for _ in range(self.numberOfChannels)]
        
        self.readTimer = QTimer()                           # Crearea unui timer care are ca scop sa colecteze datele
        self.readTimer.moveToThread(self)                   # Mutarea timer-ului in thread-ul curent
        self.readTimer.timeout.connect(self.collectData)    # Apelarea functiei de colectare si prelucrare atunci cand se termina numaratoarea
            
    def run(self):      # RUN
        self.readTimer.start(250)   # Pornirea si setarea duratei timer-ului
        loop = QEventLoop()         # Crearea unui eveniment de tip loop care mentine thread-ul deschis
        loop.exec_()                # Loop
    
    def collectData(self):  # Functia care colecteaza datele
        if self.quitFlag == True:
            print("UDP - connection closed")
            self.readTimer.stop()
            self.sock.close()
            self.quit()
            return
        
        if self.errorConnectionFlag == True:
            print("UDP - error")
            self.errorSignal.emit()
            return
        
        for _ in range(10):
            self.sock.send(self.start_cmd)
        try:
            self.buffer, addr = self.sock.recvfrom(1024)
        except socket.timeout:
            for _ in range(10):
                self.sock.send(self.retry_cmd)
            try:
                self.buffer, addr = self.sock.recvfrom(1024)
            except socket.timeout:
                print("UDP - timeout")
                self.buffer = []
                if self.counterTimeout == 8:
                    self.counterTimeout = 0
                    self.errorSignal.emit()
                self.counterTimeout += 1
        
        if(len(self.buffer) == 1024):
            
            self.buffer = memoryview(self.buffer)
            self.buffer = [int(x)-128 for x in self.buffer]
            
            current_time = time.time_ns() - self.start_moment
            delta = (current_time - self.old_time) / 128
            vect_time = np.arange(self.old_time, current_time, delta).tolist()
            self.old_time = current_time
            
            try:
                self.mutex.lock()
                self.data['x'].extend(vect_time)
                self.data['x'] = self.data['x'][128:]
                self.mutex.unlock()
                
                for channel in range(self.numberOfChannels):  # Pentru fiecare canal
                    self.mutex.lock()
                    self.data['y'][channel].extend(self.buffer[channel*128:  (channel+1)*128])
                    self.data['y'][channel] = self.data['y'][channel][128:]
                    
                    self.windows.append(self.buffer[channel*128:  (channel+1)*128])
                    self.mutex.unlock()
            except:
                pass
            
            copyWindows = self.windows.copy()
            
            #print(type(copyWindows[0][0]))
            self.readyWindowsSignal.emit(copyWindows)
            
            self.mutex.lock()
            for _ in range(self.numberOfChannels):
                self.windows.pop(0)
            self.mutex.unlock()
    
    def stop(self):
        self.quitFlag = True
