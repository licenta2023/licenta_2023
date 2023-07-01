from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal
import pyqtgraph as pg
import scipy.signal as signal
import numpy as np

    
class FIRWidget(QWidget):
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
        
        # Parametrii filtrului FIR
        self.sampling_freq = 512  # Frecvența de eșantionare în Hz
        self.filter_length = 65  # Lungimea filtrului FIR (ordinea + 1)
        self.cutoff_freq = 100
        self.bandpass_freq = [100, 200]  # Frecvențele de bandă de trecere în Hz
        self.stopband_freq = [90, 210]  # Frecvențele de bandă de oprire în Hz
        self.beta = 5  # Parametrul beta al fereastrei Kaiser
        
        # Proiectarea filtrului FIR trece-bandă folosind fereastra Kaiser
        self.filter_coeffs = signal.firwin(self.filter_length, self.cutoff_freq, fs=self.sampling_freq, window=('kaiser', self.beta), pass_zero=True)
        
        # Răspunsul în frecvență al filtrului
        self.w, self.h = signal.freqz(self.filter_coeffs)
        
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
        if settings[0] == 'FTJ':
            self.filter_length = settings[1]                    # Lungimea filtrului FIR (ordinea + 1)
            self.cutoff_freq = settings[2]                      # Frecvența de taiere în Hz
            self.window = settings[5]                           # Tipul ferestrei
            if self.window == 'kaiser':
                self.beta = settings[6]                         # Parametrul beta al fereastrei Kaiser
                self.filter_coeffs = signal.firwin(self.filter_length, self.cutoff_freq, fs=self.sampling_freq, window=('kaiser', self.beta), pass_zero=True)
            else:
                self.filter_coeffs = signal.firwin(self.filter_length, self.cutoff_freq, fs=self.sampling_freq, window=self.window, pass_zero=True)
        elif settings[0] == 'FTS':
            self.filter_length = settings[1]                    # Lungimea filtrului FIR (ordinea + 1)
            self.cutoff_freq = settings[2]                      # Frecvența de taiere în Hz
            self.window = settings[5]                           # Tipul ferestrei
            if self.window == 'kaiser':
                self.beta = settings[6]                         # Parametrul beta al fereastrei Kaiser
                self.filter_coeffs = signal.firwin(self.filter_length, self.cutoff_freq, fs=self.sampling_freq, window=('kaiser', self.beta), pass_zero=False)
            else:
                self.filter_coeffs = signal.firwin(self.filter_length, self.cutoff_freq, fs=self.sampling_freq, window=self.window, pass_zero=False)
        elif settings[0] == 'FTB':
            self.filter_length = settings[1]                    # Lungimea filtrului FIR (ordinea + 1)
            self.bandpass_freq = [settings[3], settings[4]]     # Frecvențele de bandă de trecere în Hz
            self.window = settings[5]                           # Tipul ferestrei
            if self.window == 'kaiser':
                self.beta = settings[6]                         # Parametrul beta al fereastrei Kaiser
                self.filter_coeffs = signal.firwin(self.filter_length, self.bandpass_freq, fs=self.sampling_freq, window=('kaiser', self.beta), pass_zero=False)
            else:
                self.filter_coeffs = signal.firwin(self.filter_length, self.bandpass_freq, fs=self.sampling_freq, window=self.window, pass_zero=False)
        elif settings[0] == 'FOB':
            self.filter_length = settings[1]                    # Lungimea filtrului FIR (ordinea + 1)
            self.stopband_freq = [settings[3], settings[4]]     # Frecvențele de bandă de trecere în Hz
            self.window = settings[5]                           # Tipul ferestrei
            if self.window == 'kaiser':
                self.beta = settings[6]                         # Parametrul beta al fereastrei Kaiser
                self.filter_coeffs = signal.firwin(self.filter_length, self.stopband_freq, fs=self.sampling_freq, window=('kaiser', self.beta), pass_zero=True)
            else:
                self.filter_coeffs = signal.firwin(self.filter_length, self.stopband_freq, fs=self.sampling_freq, window=self.window, pass_zero=True)
        
        # Răspunsul în frecvență al filtrului
        self.w, self.h = signal.freqz(self.filter_coeffs)
        
        self.updateplot()
