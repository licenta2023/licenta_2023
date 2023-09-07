from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal
import pyqtgraph as pg
import scipy.signal as signal
import numpy as np


class IIRWidget(QWidget):
    errorSignal = pyqtSignal()
    
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)              # Initializare widget
        pg.setConfigOption("background","w")        # Fundal alb
        
        # Initializari in interfata
        self.vLine=pg.InfiniteLine(angle=90,movable=False)
        self.hLine=pg.InfiniteLine(angle=0,movable=False)
        self.dataPosX = 0
        
        self.label = pg.LabelItem(justify="left")
        hbox = QVBoxLayout()
		
        self.setLayout(hbox)
        self.plotwidget = pg.PlotWidget()
        self.plotwidget.addItem(self.vLine,ignoreBounds=True)
        self.plotwidget.addItem(self.hLine,ignoreBounds=True)
        self.vb = self.plotwidget.plotItem.vb
        self.plotwidget.addItem(self.label)
        hbox.addWidget(self.plotwidget)
        
        self.plotcurves = pg.PlotCurveItem()
        self.plotwidget.addItem(self.plotcurves)
        
        self.pen = pg.mkPen(color='red', width=5)
        
        # Specificațiile filtrului
        self.cutoff_freq = [100, 150]  # Frecvențele de bandă de trecere în Hz
        self.stopband_freq = [90, 160]  # Frecvențele de bandă de oprire în Hz
        self.sampling_freq = 512  # Frecvența de eșantionare în Hz
        self.ripple_attenuation = 1  # Atenuarea maximă admisă în banda de trecere în dB
        self.stopband_attenuation = 40  # Atenuarea minimă dorită în banda de oprire în dB
        self.order = 4  # Ordinea filtrului Butterworth
        
        # Proiectarea filtrului FIR trece-bandă folosind fereastra Kaiser
        self.order, self.wn = signal.ellipord(self.cutoff_freq, self.stopband_freq, self.ripple_attenuation, self.stopband_attenuation, fs=self.sampling_freq)
        self.b, self.a = signal.ellip(self.order, self.ripple_attenuation, self.stopband_attenuation, self.wn, btype='band', fs=self.sampling_freq)
        
        # Răspunsul în frecvență al filtrului
        self.w, self.h = signal.freqz(self.b, self.a)
        
        self.updateplot()
        
        self.plotwidget.setRange(yRange=[-0.1, max(np.abs(self.h))+0.1]) 
        
        self.curvePoints = pg.CurvePoint(self.plotcurves)
        self.plotwidget.addItem(self.curvePoints)
        
        self.texts = pg.TextItem(str(1), color='red', anchor=(0.5,-0.5))
        self.texts.setParentItem(self.curvePoints)
        
        self.proxy = pg.SignalProxy (self.plotwidget.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
            
    def mouseMoved(self, evt):
        pos = evt[0] 
		
        if self.plotwidget.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            scale = 0.5 * self.sampling_freq * (max(self.w) - min(self.w)) / np.pi
            w_scaled = (mousePoint.x() - (0.5 * self.sampling_freq * min(self.w) / np.pi)) / scale
            index = int(w_scaled * len(self.h))
            
            if index >=0 and index < len(self.w):
                dataPosX = mousePoint.x()
				
                self.curvePoints.setPos(float(index)/(len(self.w)-1))
                self.texts.setText("%0.3f, %0.3f"%(dataPosX, np.abs(self.h[index])))
			
                self.vLine.setPos(mousePoint.x())
                self.hLine.setPos(np.abs(self.h[index]))
    
    def updateplot(self):
        self.plotcurves.setData(0.5 * self.sampling_freq * self.w / np.pi, np.abs(self.h), pen=self.pen, clickable=True)

    def applyNewSettings(self, settings):
        self.cutoff_freq = [settings[2], settings[3]]  # Frecvențele de bandă de trecere în Hz
        self.stopband_freq = [settings[4], settings[5]]  # Frecvențele de bandă de oprire în Hz
        self.ripple_attenuation = settings[6]  # Atenuarea maximă admisă în banda de trecere în dB
        self.stopband_attenuation = settings[7]  # Atenuarea minimă dorită în banda de oprire în dB
        self.order = settings[8]  # Ordinul filtrului Butterworth
        
        if settings[0] == 'Butterworth':
            if settings[1] == 'FTJ':
                self.b, self.a = signal.butter(self.order, self.cutoff_freq[0], btype='low', fs=self.sampling_freq)
            elif settings[1] == 'FTS':
                self.b, self.a = signal.butter(self.order, self.cutoff_freq[0], btype='high', fs=self.sampling_freq)
            elif settings[1] == 'FTB':
                self.b, self.a = signal.butter(self.order, self.cutoff_freq, btype='band', fs=self.sampling_freq)
            elif settings[1] == 'FOB':
                self.b, self.a = signal.butter(self.order, self.cutoff_freq, btype='bandstop', fs=self.sampling_freq)
        elif settings[0] == 'Cebisev I':
            if settings[1] == 'FTJ':
                self.order, self.wn = signal.cheb1ord(self.cutoff_freq[0], self.stopband_freq[0], self.ripple_attenuation, self.stopband_attenuation, fs=self.sampling_freq)
                self.b, self.a = signal.cheby1(self.order, self.ripple_attenuation, self.wn, btype='low', fs=self.sampling_freq)
            elif settings[1] == 'FTS':
                self.order, self.wn = signal.cheb1ord(self.cutoff_freq[0], self.stopband_freq[0], self.ripple_attenuation, self.stopband_attenuation, fs=self.sampling_freq)
                self.b, self.a = signal.cheby1(self.order, self.ripple_attenuation, self.wn, btype='high', fs=self.sampling_freq)
            elif settings[1] == 'FTB':
                self.order, self.wn = signal.cheb1ord(self.cutoff_freq, self.stopband_freq, self.ripple_attenuation, self.stopband_attenuation, fs=self.sampling_freq)
                self.b, self.a = signal.cheby1(self.order, self.ripple_attenuation, self.wn, btype='band', fs=self.sampling_freq)
            elif settings[1] == 'FOB':
                self.order, self.wn = signal.cheb1ord(self.cutoff_freq, self.stopband_freq, self.ripple_attenuation, self.stopband_attenuation, fs=self.sampling_freq)
                self.b, self.a = signal.cheby1(self.order, self.ripple_attenuation, self.wn, btype='bandstop', fs=self.sampling_freq)
        elif settings[0] == 'Cebisev II':
            if settings[1] == 'FTJ':
                self.order, self.wn = signal.cheb2ord(self.cutoff_freq[0], self.stopband_freq[0], self.ripple_attenuation, self.stopband_attenuation, fs=self.sampling_freq)
                self.b, self.a = signal.cheby2(self.order, self.stopband_attenuation, self.wn, btype='low', fs=self.sampling_freq)
            elif settings[1] == 'FTS':
                self.order, self.wn = signal.cheb2ord(self.cutoff_freq[0], self.stopband_freq[0], self.ripple_attenuation, self.stopband_attenuation, fs=self.sampling_freq)
                self.b, self.a = signal.cheby2(self.order, self.stopband_attenuation, self.wn, btype='high', fs=self.sampling_freq)
            elif settings[1] == 'FTB':
                self.order, self.wn = signal.cheb2ord(self.cutoff_freq, self.stopband_freq, self.ripple_attenuation, self.stopband_attenuation, fs=self.sampling_freq)
                self.b, self.a = signal.cheby2(self.order, self.stopband_attenuation, self.wn, btype='band', fs=self.sampling_freq)
            elif settings[1] == 'FOB':
                self.order, self.wn = signal.cheb2ord(self.cutoff_freq, self.stopband_freq, self.ripple_attenuation, self.stopband_attenuation, fs=self.sampling_freq)
                self.b, self.a = signal.cheby2(self.order, self.stopband_attenuation, self.wn, btype='bandstop', fs=self.sampling_freq)
        elif settings[0] == 'Eliptic':
            if settings[1] == 'FTJ':
                self.order, self.wn = signal.ellipord(self.cutoff_freq[0], self.stopband_freq[0], self.ripple_attenuation, self.stopband_attenuation, fs=self.sampling_freq)
                self.b, self.a = signal.ellip(self.order, self.ripple_attenuation, self.stopband_attenuation, self.wn, btype='low', fs=self.sampling_freq)
            elif settings[1] == 'FTS':
                self.order, self.wn = signal.ellipord(self.cutoff_freq[0], self.stopband_freq[0], self.ripple_attenuation, self.stopband_attenuation, fs=self.sampling_freq)
                self.b, self.a = signal.ellip(self.order, self.ripple_attenuation, self.stopband_attenuation, self.wn, btype='high', fs=self.sampling_freq)
            elif settings[1] == 'FTB':
                self.order, self.wn = signal.ellipord(self.cutoff_freq, self.stopband_freq, self.ripple_attenuation, self.stopband_attenuation, fs=self.sampling_freq)
                self.b, self.a = signal.ellip(self.order, self.ripple_attenuation, self.stopband_attenuation, self.wn, btype='band', fs=self.sampling_freq)
            elif settings[1] == 'FOB':
                self.order, self.wn = signal.ellipord(self.cutoff_freq, self.stopband_freq, self.ripple_attenuation, self.stopband_attenuation, fs=self.sampling_freq)
                self.b, self.a = signal.ellip(self.order, self.ripple_attenuation, self.stopband_attenuation, self.wn, btype='bandstop', fs=self.sampling_freq)
        
        # Răspunsul în frecvență al filtrului
        self.w, self.h = signal.freqz(self.b, self.a)
        
        self.updateplot()
