from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtSerialPort import QSerialPort
from PyQt5.QtCore import Qt, Q_ARG, QMetaObject
from PyQt5.uic import loadUi
from pyqtgraph.opengl import MeshData, GLMeshItem
from stl import mesh

import sys
import numpy as np

from utility.device import Device
from utility.read_usb import ReadPort
from utility.read_udp import ReadUDP
from utility.plot import Plot
from utility.record import Record
from utility.processing import Processing
from utility.ml import ML

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        loadUi("interface/mainwindow.ui", self)
        self.frame_conectare.close()
        self.frame_prelucrare.close()
        self.frame_afisare.close()
        self.frame_gesturi.close()
        
        self.device = None
        self.read = None
        
        self.initConectare()
        self.initPrelucrare()
        self.initAfisare()
        self.initInterpretareGesturi()
    
    def initConectare(self):
        self.dataRead = {'x': None, 'y': None}
        self.numberOfChannels = 8
        
        self.device = Device()
        self.device.updateSignal.connect(self.updatePortNameComboBox)
        
        self.baudRateComboBox.addItem('9600')
        self.baudRateComboBox.addItem('14400')
        self.baudRateComboBox.addItem('28800')
        self.baudRateComboBox.addItem('57600')
        self.baudRateComboBox.addItem('115200')
        self.baudRateComboBox.addItem('921600')
        
        self.dataBitsComboBox.addItem('6')
        self.dataBitsComboBox.addItem('7')
        self.dataBitsComboBox.addItem('8')
        
        self.parityComboBox.addItem('NoParity')
        self.parityComboBox.addItem('Even')
        self.parityComboBox.addItem('Odd')
        
        self.stopBitsComboBox.addItem('1')
        self.stopBitsComboBox.addItem('2')
        
        self.openStateUSB = False
        self.openStateUDP = False
        self.openUSBPushButton.clicked.connect(self.togglePortUSB)
        self.openUDPPushButton.clicked.connect(self.togglePortUDP)
        
    
    def initPrelucrare(self):        
        # Initializari in fereastra de prelucrare
        self.comboBox_tipFiltruFIR.addItem('FTJ')
        self.comboBox_tipFiltruFIR.addItem('FTS')
        self.comboBox_tipFiltruFIR.addItem('FTB')
        self.comboBox_tipFiltruFIR.addItem('FOB')
        
        self.lineEdit_ordinFIR.setText('65')
        
        self.lineEdit_frecTaiereFIR.setText('100')
        
        self.lineEdit_bandaStartFIR.setText('100')
        self.lineEdit_bandaStopFIR.setText('150')
        
        self.comboBox_tipFereastraFIR.addItem('hamming')
        self.comboBox_tipFereastraFIR.addItem('hanning')
        self.comboBox_tipFereastraFIR.addItem('boxcar')
        self.comboBox_tipFereastraFIR.addItem('blackman')
        self.comboBox_tipFereastraFIR.addItem('kaiser')
        
        self.lineEdit_betaFIR.setText('7.865')
        
        self.settingsFIR = [self.comboBox_tipFiltruFIR.currentText(), int(self.lineEdit_ordinFIR.text()),
                            float(self.lineEdit_frecTaiereFIR.text()), float(self.lineEdit_bandaStartFIR.text()),
                            float(self.lineEdit_bandaStopFIR.text()), self.comboBox_tipFereastraFIR.currentText(),
                            float(self.lineEdit_betaFIR.text())]
        
        self.pushButton_aplicaFIR.clicked.connect(self.applySettingsAndPlotFIR)
        
        
        self.comboBox_FiltruIIR.addItem('Eliptic')
        self.comboBox_FiltruIIR.addItem('Cebisev I')
        self.comboBox_FiltruIIR.addItem('Cebisev II')
        self.comboBox_FiltruIIR.addItem('Butterworth')
        
        self.comboBox_tipFiltruIIR.addItem('FTJ')
        self.comboBox_tipFiltruIIR.addItem('FTS')
        self.comboBox_tipFiltruIIR.addItem('FTB')
        self.comboBox_tipFiltruIIR.addItem('FOB')
        
        self.lineEdit_frecTaiere1IIR.setText('100')
        self.lineEdit_frecTaiere2IIR.setText('150')
        
        self.lineEdit_frecStop1IIR.setText('90')
        self.lineEdit_frecStop2IIR.setText('160')
        
        self.lineEdit_atenuareRippleIIR.setText('1')
        
        self.lineEdit_atenuareStopBandIIR.setText('40')
        
        self.lineEdit_ordinIIR.setText('4')
        
        self.settingsIIR = [self.comboBox_FiltruIIR.currentText(), self.comboBox_tipFiltruIIR.currentText(),
                            float(self.lineEdit_frecTaiere1IIR.text()), float(self.lineEdit_frecTaiere2IIR.text()),
                            float(self.lineEdit_frecStop1IIR.text()), float(self.lineEdit_frecStop2IIR.text()),
                            float(self.lineEdit_atenuareRippleIIR.text()), float(self.lineEdit_atenuareStopBandIIR.text()),
                            int(self.lineEdit_ordinIIR.text())]
        
        self.pushButton_aplicaIIR.clicked.connect(self.applySettingsAndPlotIIR)
    
    def initAfisare(self):
        self.processedData = {'x': [None]*self.numberOfChannels, 'y': [None]*self.numberOfChannels}
        self.settings = [None] * self.numberOfChannels
        
        self.processingThread = Processing(self.dataRead, self.processedData, self.settings, self.numberOfChannels)
        self.processingThread.start()
        
        # Lista widget-urilor de tip plot din interfata
        pltWidgets = [self.plotWidget_channel_1, self.plotWidget_channel_2,
                      self.plotWidget_channel_3, self.plotWidget_channel_4, 
                      self.plotWidget_channel_5, self.plotWidget_channel_6,
                      self.plotWidget_channel_7, self.plotWidget_channel_8]
        
        # Pornirea thread-ului care ploteaza
        self.plots = Plot(self.processedData, self.numberOfChannels, pltWidgets)
        self.plots.start()
        
        # Conexiuni pentru butoane
        self.fftAbsButtons = [self.fftAbsChannel1pushButton, self.fftAbsChannel2pushButton,
                              self.fftAbsChannel3pushButton, self.fftAbsChannel4pushButton,
                              self.fftAbsChannel5pushButton, self.fftAbsChannel6pushButton,
                              self.fftAbsChannel7pushButton, self.fftAbsChannel8pushButton]
        
        self.fftPhaseButtons = [self.fftPhaseChannel1pushButton, self.fftPhaseChannel2pushButton,
                                self.fftPhaseChannel3pushButton, self.fftPhaseChannel4pushButton,
                                self.fftPhaseChannel5pushButton, self.fftPhaseChannel6pushButton,
                                self.fftPhaseChannel7pushButton, self.fftPhaseChannel8pushButton]
        
        self.firButtons = [self.pushButton_FIR_1, self.pushButton_FIR_2,
                           self.pushButton_FIR_3, self.pushButton_FIR_4,
                           self.pushButton_FIR_5, self.pushButton_FIR_6,
                           self.pushButton_FIR_7, self.pushButton_FIR_8]
        
        self.iirButtons = [self.pushButton_IIR_1, self.pushButton_IIR_2,
                           self.pushButton_IIR_3, self.pushButton_IIR_4,
                           self.pushButton_IIR_5, self.pushButton_IIR_6,
                           self.pushButton_IIR_7, self.pushButton_IIR_8]
        
        self.fftAbsChannel1pushButton.clicked.connect(lambda: self.fftAbsSetting(0))
        self.fftAbsChannel2pushButton.clicked.connect(lambda: self.fftAbsSetting(1))
        self.fftAbsChannel3pushButton.clicked.connect(lambda: self.fftAbsSetting(2))
        self.fftAbsChannel4pushButton.clicked.connect(lambda: self.fftAbsSetting(3))
        self.fftAbsChannel5pushButton.clicked.connect(lambda: self.fftAbsSetting(4))
        self.fftAbsChannel6pushButton.clicked.connect(lambda: self.fftAbsSetting(5))
        self.fftAbsChannel7pushButton.clicked.connect(lambda: self.fftAbsSetting(6))
        self.fftAbsChannel8pushButton.clicked.connect(lambda: self.fftAbsSetting(7))
        
        self.fftPhaseChannel1pushButton.clicked.connect(lambda: self.fftPhaseSetting(0))
        self.fftPhaseChannel2pushButton.clicked.connect(lambda: self.fftPhaseSetting(1))
        self.fftPhaseChannel3pushButton.clicked.connect(lambda: self.fftPhaseSetting(2))
        self.fftPhaseChannel4pushButton.clicked.connect(lambda: self.fftPhaseSetting(3))
        self.fftPhaseChannel5pushButton.clicked.connect(lambda: self.fftPhaseSetting(4))
        self.fftPhaseChannel6pushButton.clicked.connect(lambda: self.fftPhaseSetting(5))
        self.fftPhaseChannel7pushButton.clicked.connect(lambda: self.fftPhaseSetting(6))
        self.fftPhaseChannel8pushButton.clicked.connect(lambda: self.fftPhaseSetting(7))
        
        self.pushButton_FIR_1.clicked.connect(lambda: self.firSetting(0))
        self.pushButton_FIR_2.clicked.connect(lambda: self.firSetting(1))
        self.pushButton_FIR_3.clicked.connect(lambda: self.firSetting(2))
        self.pushButton_FIR_4.clicked.connect(lambda: self.firSetting(3))
        self.pushButton_FIR_5.clicked.connect(lambda: self.firSetting(4))
        self.pushButton_FIR_6.clicked.connect(lambda: self.firSetting(5))
        self.pushButton_FIR_7.clicked.connect(lambda: self.firSetting(6))
        self.pushButton_FIR_8.clicked.connect(lambda: self.firSetting(7))
        
        self.pushButton_IIR_1.clicked.connect(lambda: self.iirSetting(0))
        self.pushButton_IIR_2.clicked.connect(lambda: self.iirSetting(1))
        self.pushButton_IIR_3.clicked.connect(lambda: self.iirSetting(2))
        self.pushButton_IIR_4.clicked.connect(lambda: self.iirSetting(3))
        self.pushButton_IIR_5.clicked.connect(lambda: self.iirSetting(4))
        self.pushButton_IIR_6.clicked.connect(lambda: self.iirSetting(5))
        self.pushButton_IIR_7.clicked.connect(lambda: self.iirSetting(6))
        self.pushButton_IIR_8.clicked.connect(lambda: self.iirSetting(7))
        
        # Alte initializari
        self.comboBox_gesture_number.addItem('0')
        self.comboBox_gesture_number.addItem('1')
        self.comboBox_gesture_number.addItem('2')
        self.comboBox_gesture_number.addItem('3')
        self.comboBox_gesture_number.addItem('4')
        self.comboBox_gesture_number.addItem('5')
        self.comboBox_gesture_number.addItem('6')
        self.comboBox_gesture_number.addItem('7')
        self.comboBox_gesture_number.addItem('8')
        self.comboBox_gesture_number.addItem('9')
        self.comboBox_gesture_number.addItem('10')
        self.comboBox_gesture_number.addItem('11')
        self.comboBox_gesture_number.addItem('12')
        self.comboBox_gesture_number.addItem('13')
        self.comboBox_gesture_number.addItem('14')
        
        self.recordingFlag = False
        self.button_mem.clicked.connect(self.recordGesture)
    
    def initInterpretareGesturi(self):
        # Pornirea thread-ului care identifica gesturile facute
        self.gestureNumber = 0
        self.classificationFlag = True
        self.NeuralNetwork = ML(self.numberOfChannels)
        self.NeuralNetwork.doneSignal.connect(self.classificationDone)
        self.NeuralNetwork.start()
        
        self.comboBox_NN.addItem('NN - 5 gesturi - caracteristici in timp - TensorFlow')
        self.comboBox_NN.addItem('NN - 5 gesturi - caracteristici in frecventa - TensorFlow')
        self.comboBox_NN.addItem('NN - 5 gesturi - caracteristici in timp - Versiunea mea')
        self.comboBox_NN.addItem('NN - 5 gesturi - caracteristici in frecventa - Versiunea mea')
        self.comboBox_NN.addItem('NN - 15 gesturi - caracteristici in timp - TensorFlow')
        self.comboBox_NN.addItem('NN - 15 gesturi - caracteristici in frecventa - TensorFlow')
        self.comboBox_NN.addItem('NN - 15 gesturi - caracteristici in timp - Versiunea mea')
        self.comboBox_NN.addItem('NN - 15 gesturi - caracteristici in frecventa - Versiunea mea')
        
        self.move = [self.moveNeutralPosition, self.moveRadialDeviation, self.moveWristFlexion,
                     self.moveUlnarDeviation, self.moveWristExtention, self.moveFingersFlexion,
                     self.moveFingersExtention, self.moveLetterI, self.moveLetterD,
                     self.moveLetterV, self.moveLetterF, self.moveLetterL,
                     self.moveLetterY, self.moveLike, self.moveGrab]
        
        self.mesh = [None] * 16
        self.currentMesh = None
        
        stl = mesh.Mesh.from_file('./hand/idle.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[0] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/neutral_position.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[1] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/radial_deviation.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[2] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/wrist_flexion.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[3] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/ulnar_deviation.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[4] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/wrist_extention.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[5] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/fingers_flexion.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[6] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/fingers_extention.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[7] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/letter_I.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[8] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/letter_D.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[9] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/letter_V.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[10] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/letter_F.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[11] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/letter_L.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[12] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/letter_Y.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[13] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/like.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[14] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        stl = mesh.Mesh.from_file('./hand/grab.stl')
        points = stl.points.reshape(-1, 3)
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        mesh_data = MeshData(vertexes=points, faces=faces)
        self.mesh[15] = GLMeshItem(meshdata=mesh_data, smooth=True, drawFaces=True, drawEdges=False, shader='viewNormalColor', glOptions='opaque')
        
        # Crearea unui widget GLViewWidget si adaugarea mesh-ului
        self.widget_model.setBackgroundColor(255, 255, 255)
        self.widget_model.setCameraPosition(distance=400, elevation=0, azimuth=180)
        self.currentMesh = self.mesh[0]
        self.widget_model.addItem(self.currentMesh)
    
    def closeEvent(self, event):
        if self.openStateUDP == True:
            self.closeWirelessConnection()
            
        if self.openStateUSB == True:
            self.closePort()
        
        self.device.close()
        
        self.NeuralNetwork.close()
        self.plots.close()
        
        event.accept()
        QApplication.quit()
    
    def recordGesture(self):
        gestureNr = self.comboBox_gesture_number.currentText()
        
        self.rec = Record(self.numberOfChannels, gestureNr)
        self.rec.doneSignal.connect(self.recordingDone)
        self.rec.errorSignal.connect(self.recordingError)
        
        self.rec.start()
        self.recordingFlag = True
        self.button_mem.setEnabled(False)
        self.button_mem.setStyleSheet("background-color: lightgray;")
    
    def recordingDone(self):
        self.recordingFlag = False
        self.button_mem.setEnabled(True)
        self.button_mem.setStyleSheet("background-color: lightgreen;")
    
    def recordingError(self):
        self.recordingFlag = False
        self.button_mem.setEnabled(True)
        self.button_mem.setStyleSheet("background-color: red;")
    
    def togglePortUDP(self):
        if self.openStateUSB == True:
            self.closePort()
        
        if self.openStateUDP == False:
            self.openWirelessConnection()
        else:
            self.closeWirelessConnection()
    
    def togglePortUSB(self):
        if self.openStateUDP == True:
            self.closeWirelessConnection()
        
        if self.openStateUSB == False:
            self.openPort()
        else:
            self.closePort()
    
    def openWirelessConnection(self):
        self.openStateUDP = True
        self.openUDPPushButton.setText("Stop")
        
        self.windows = []
        self.read = ReadUDP(self.dataRead, self.windows, self.numberOfChannels)
        self.read.readyWindowsSignal.connect(self.readyWindows)
        self.read.errorSignal.connect(self.closeWirelessConnection)
        
        self.read.start()
    
    def closeWirelessConnection(self):
        self.openStateUDP = False
        self.openUDPPushButton.setText("Start")
        
        self.read.stop()
        self.read.wait()
        
        self.classificationFlag = True
    
    def openPort(self):
        self.openStateUSB = True
        self.openUSBPushButton.setText("Stop")
    
        name = self.portNameComboBox.currentText()
        baud = int(self.baudRateComboBox.currentText())
        dataBits = int(self.dataBitsComboBox.currentText())
        parity = self.parityComboBox.currentText()
        stopBits = int(self.stopBitsComboBox.currentText())
    
        if parity == 'NoParity':
            parity = QSerialPort.NoParity
        elif parity == 'Even':
            parity = QSerialPort.EvenParity
        elif parity == 'Odd':
            parity = QSerialPort.OddParity
    
        self.device.setParameters(name, baud, dataBits, parity, stopBits)
        self.device.portOpen(True)
        self.device.port.write("#g".encode())
    
        self.windows = []
        self.read = ReadPort(self.dataRead, self.windows, self.device, self.numberOfChannels)
        self.read.readyWindowsSignal.connect(self.readyWindows)
        self.read.errorSignal.connect(self.closePort)
        
        self.read.start()
    
    def closePort(self):
        self.openStateUSB = False
        self.openUSBPushButton.setText("Start")
        
        self.read.stop()
        self.read.wait()
    
        self.device.port.write("#h".encode())
        self.device.portOpen(False)
        
        self.classificationFlag = True
    
    def updatePortNameComboBox(self, portList):
        self.portNameComboBox.clear()
        for port in portList:
            self.portNameComboBox.addItem(port)
    
    def readyWindows(self, window):
        arg = [self.plotFIR.filter_coeffs.tolist(), self.plotIIR.a.tolist(), self.plotIIR.b.tolist()]
        QMetaObject.invokeMethod(self.processingThread, 'process', Qt.QueuedConnection, Q_ARG(list, arg))
        
        if self.recordingFlag == True:
            QMetaObject.invokeMethod(self.rec, 'record', Qt.QueuedConnection, Q_ARG(list, window))
        
        if self.classificationFlag == True:
            self.classificationFlag = False
            arg = [window, self.comboBox_NN.currentText()]
            QMetaObject.invokeMethod(self.NeuralNetwork, 'classificationTime', Qt.QueuedConnection, Q_ARG(list, arg))
    
    def fftAbsSetting(self, channel):
        if self.fftAbsButtons[channel].isChecked():
            self.settings[channel] = 'fftAbs'
            
            self.fftPhaseButtons[channel].setChecked(False)
            self.fftPhaseButtons[channel].setStyleSheet("background-color: white;")
            self.firButtons[channel].setChecked(False)
            self.firButtons[channel].setStyleSheet("background-color: white;")
            self.iirButtons[channel].setChecked(False)
            self.iirButtons[channel].setStyleSheet("background-color: white;")
            
            self.fftAbsButtons[channel].setStyleSheet("background-color: lightgray;")
        else:
            self.settings[channel] = None
            self.fftAbsButtons[channel].setStyleSheet("background-color: white;")
    
    def fftPhaseSetting(self, channel):
        if self.fftPhaseButtons[channel].isChecked():
            self.settings[channel] = 'fftPhase'
            
            self.fftAbsButtons[channel].setChecked(False)
            self.fftAbsButtons[channel].setStyleSheet("background-color: white;")
            self.firButtons[channel].setChecked(False)
            self.firButtons[channel].setStyleSheet("background-color: white;")
            self.iirButtons[channel].setChecked(False)
            self.iirButtons[channel].setStyleSheet("background-color: white;")
            
            self.fftPhaseButtons[channel].setStyleSheet("background-color: lightgray;")
        else:
            self.settings[channel] = None
            self.fftPhaseButtons[channel].setStyleSheet("background-color: white;")
    
    def firSetting(self, channel):
        if self.firButtons[channel].isChecked():
            self.settings[channel] = 'fir'
            
            self.fftAbsButtons[channel].setChecked(False)
            self.fftAbsButtons[channel].setStyleSheet("background-color: white;")
            self.fftPhaseButtons[channel].setChecked(False)
            self.fftPhaseButtons[channel].setStyleSheet("background-color: white;")
            self.iirButtons[channel].setChecked(False)
            self.iirButtons[channel].setStyleSheet("background-color: white;")
            
            self.firButtons[channel].setStyleSheet("background-color: lightgray;")
        else:
            self.settings[channel] = None
            self.firButtons[channel].setStyleSheet("background-color: white;")
    
    def iirSetting(self, channel):
        if self.iirButtons[channel].isChecked():
            self.settings[channel] = 'iir'
            
            self.fftAbsButtons[channel].setChecked(False)
            self.fftAbsButtons[channel].setStyleSheet("background-color: white;")
            self.fftPhaseButtons[channel].setChecked(False)
            self.fftPhaseButtons[channel].setStyleSheet("background-color: white;")
            self.firButtons[channel].setChecked(False)
            self.firButtons[channel].setStyleSheet("background-color: white;")
            
            self.iirButtons[channel].setStyleSheet("background-color: lightgray;")
        else:
            self.settings[channel] = None
            self.iirButtons[channel].setStyleSheet("background-color: white;")
    
    def classificationDone(self, classification):
        self.gestureNumber = classification
        #print(self.gestureNumber)
        self.classificationFlag = True
        self.move[self.gestureNumber]()
    
    def applySettingsAndPlotFIR(self):
        self.settingsFIR = [self.comboBox_tipFiltruFIR.currentText(), int(self.lineEdit_ordinFIR.text()),
                            float(self.lineEdit_frecTaiereFIR.text()), float(self.lineEdit_bandaStartFIR.text()),
                            float(self.lineEdit_bandaStopFIR.text()), self.comboBox_tipFereastraFIR.currentText(),
                            float(self.lineEdit_betaFIR.text())]
        self.plotFIR.applyNewSettings(self.settingsFIR)
    
    def applySettingsAndPlotIIR(self):
        self.settingsIIR = [self.comboBox_FiltruIIR.currentText(), self.comboBox_tipFiltruIIR.currentText(),
                            float(self.lineEdit_frecTaiere1IIR.text()), float(self.lineEdit_frecTaiere2IIR.text()),
                            float(self.lineEdit_frecStop1IIR.text()), float(self.lineEdit_frecStop2IIR.text()),
                            float(self.lineEdit_atenuareRippleIIR.text()), float(self.lineEdit_atenuareStopBandIIR.text()),
                            int(self.lineEdit_ordinIIR.text())]
        self.plotIIR.applyNewSettings(self.settingsIIR)
        
    def moveIdle(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[0]
        self.widget_model.addItem(self.currentMesh)

    def moveNeutralPosition(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[1]
        self.widget_model.addItem(self.currentMesh)
    
    def moveRadialDeviation(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[2]
        self.widget_model.addItem(self.currentMesh)
        
    def moveWristFlexion(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[3]
        self.widget_model.addItem(self.currentMesh)

    def moveUlnarDeviation(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[4]
        self.widget_model.addItem(self.currentMesh)
    
    def moveWristExtention(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[5]
        self.widget_model.addItem(self.currentMesh)
    
    def moveFingersFlexion(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[6]
        self.widget_model.addItem(self.currentMesh)
    
    def moveFingersExtention(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[7]
        self.widget_model.addItem(self.currentMesh)
    
    def moveLetterI(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[8]
        self.widget_model.addItem(self.currentMesh)
    
    def moveLetterD(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[9]
        self.widget_model.addItem(self.currentMesh)
    
    def moveLetterV(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[10]
        self.widget_model.addItem(self.currentMesh)
    
    def moveLetterF(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[11]
        self.widget_model.addItem(self.currentMesh)
    
    def moveLetterL(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[12]
        self.widget_model.addItem(self.currentMesh)
    
    def moveLetterY(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[13]
        self.widget_model.addItem(self.currentMesh)
    
    def moveLike(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[14]
        self.widget_model.addItem(self.currentMesh)
    
    def moveGrab(self):
        self.widget_model.removeItem(self.currentMesh)
        self.currentMesh = self.mesh[15]
        self.widget_model.addItem(self.currentMesh)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
