from PyQt5 import QtCore
from PyQt5.QtCore import QObject
from PyQt5.QtSerialPort import QSerialPort, QSerialPortInfo
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import pyqtSignal


class Device(QObject):
    updateSignal = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.port = QSerialPort()       # Creare obiect de tip port
        self.portList = []              # Initializarea listei care contine porturile disponibile
        
        self.timer = QTimer()                       # Crearea unui timer care are ca scop sa updateze lista de porturi
        self.timer.timeout.connect(self.update)     # Apelarea functiei de updatare atunci cand se termina numaratoarea
        self.timer.start(2000)                      # Pornirea si setarea duratei timer-ului
    
    def setParameters(self, name, baud, dataBits, parity, stopBits):    # Functie pentru setarea parametrilor portului
        self.port.setPortName(name)         # Setarea numelui portului
        self.port.setBaudRate(baud)         # Setarea ratei de transmisie (baud-rate)
        self.port.setDataBits(dataBits)     # Setarea dimensiunii cuvintelor
        self.port.setParity(parity)         # Setare biti de paritate
        self.port.setStopBits(stopBits)     # Setarea numarului de biti de stop
        self.port.setFlowControl(0)         # Setarea controlului

    def portOpen(self, flag):   # Functie pentru deschiderea / inchiderea portului
        if flag:    # Daca se vrea deschiderea portului:
            openFlag = self.port.open(QtCore.QIODevice.ReadWrite)   # Deschide portul
            if not openFlag:                        # Daca nu s-a reusit deschiderea
                print("USB - connection failed")    # Afiseaza eroare
            else:                                   # Altfel
                print("USB - connected")            # Afiseaza deschis
        else:                                       # Daca se vrea inchiderea atunci:
            self.port.close()                       # Inchide
            print("USB - connection closed")        # Afiseaza inchis
    
    def update(self):   # Functie pentru updatarea listei de porturi disponibile
        self.portList.clear()                           # Se sterge lista
        for p in QSerialPortInfo().availablePorts():    # Se parcurg porturile disponibile
            self.portList.append(p.portName())          # Se adauga in lista
        print("Ports: ", self.portList)                            # Se afiseaza lista
        self.updateSignal.emit(self.portList)

    def close(self):
        self.timer.stop()
