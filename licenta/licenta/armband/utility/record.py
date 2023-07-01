from PyQt5.QtCore import QEventLoop
from PyQt5.QtCore import QThread
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import QMutex

import os
import numpy as np


class Record(QThread):
    doneSignal = pyqtSignal()
    errorSignal = pyqtSignal()
    
    def __init__(self, numberOfChannels, gestureNumber):
        QThread.__init__(self)                      # Initializare thread
        self.moveToThread(self)                     # Mut toate metodele pe thread
        self.quitFlag = False
        self.mutex = QMutex()
        self.counter = 0
        
        self.numberOfChannels = numberOfChannels
        self.gestureNumber = int(gestureNumber)
        
        self.folder_path = './ml/record' # calea catre folderul cu date
        self.data = [[] for _ in range(self.numberOfChannels)]
        
    def run(self):      # RUN
        loop = QEventLoop()         # Crearea unui eveniment de tip loop care mentine thread-ul deschis
        loop.exec_()                # Loop
    
    @pyqtSlot(list)
    def record(self, windows):
        #if QThread.currentThread() == self.thread():
        #    print("Running on a separate thread")
        #else:
        #    print("Not running on a separate thread")
        print('recording')
        if self.quitFlag == True:
            self.quit()
            return
        
        # Verific daca inregistrarea s-a terminat
        if self.counter == 20:
            self.save()
            self.quitFlag = True
            return
        
        # Adaug noile ferestre la inregistrare
        self.counter += 1
        self.mutex.lock()
        for channel in range(self.numberOfChannels):
            try:
                aux = [np.uint8(windows[channel][index] + 128) for index in range(len(windows[channel]))]
                self.data[channel].extend(aux)
            except:
                pass
        self.mutex.unlock()
    
    def save(self):
        # Verific daca au aparut erori in timpul inregistrarii
        for channel in range(self.numberOfChannels):
            if len(self.data[channel]) != self.counter*128:
                print(len(self.data[channel]), channel)
                self.errorSignal.emit()
                return
        
        # Salvez inregistrarea intr-un fisier
        # Identific ce denumire va avea fisierul
        file_list = os.listdir(self.folder_path) # crearea unei liste cu toate fisierele din acel folder
        npy_files = [file for file in file_list if file.endswith('.npy') and self.gestureNumber == int(file[3:5])] # selectare doar fisiere .npy
        
        if len(npy_files) == 0: # Daca nu exista fisiere ale gestului atunci se face primul
            fileName = '00_' + str(self.gestureNumber).zfill(2) + '_A'
        elif len(npy_files) == 1: # Daca exista un singur fisier al gestului atunci
            if npy_files[0][6:7] == 'A': # Daca e varianta A se adauga varianta B
                fileName = str(int(npy_files[0][0:2])).zfill(2) + '_' + str(self.gestureNumber).zfill(2) + '_B'
            elif npy_files[0][6:7] == 'B': # Daca e varianta B se adauga varianta A
                fileName = str(int(npy_files[0][0:2])).zfill(2) + '_' + str(self.gestureNumber).zfill(2) + '_A'
        else: # Daca sunt mai multe atunci
            npy_files = sorted(npy_files) # Se sorteaza
            if int(npy_files[-1][0:2]) == int(npy_files[-2][0:2]): # Daca exista deja 
                fileName = str(int(npy_files[-1][0:2])+1).zfill(2) + '_' + str(self.gestureNumber).zfill(2) + '_A'
            elif npy_files[-1][6:7] == 'A':
                fileName = str(int(npy_files[-1][0:2])).zfill(2) + '_' + str(self.gestureNumber).zfill(2) + '_B'
            elif npy_files[-1][6:7] == 'B':
                fileName = str(int(npy_files[-1][0:2])).zfill(2) + '_' + str(self.gestureNumber).zfill(2) + '_A'
        
        file_path = os.path.join(self.folder_path, fileName)
        np.save(file_path, self.data)
        self.doneSignal.emit()
    
    def stop(self):
        self.quitFlag = True
