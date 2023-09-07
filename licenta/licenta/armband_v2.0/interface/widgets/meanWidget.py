from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import pyqtSignal
import pyqtgraph as pg
import numpy as np

    
class MeanWidget(QWidget):
    errorSignal = pyqtSignal()
    
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)              # Initializare widget
        pg.setConfigOption("background","w")        # Fundal alb
        
        # Initializari in interfata
        self.vLine=pg.InfiniteLine(angle=90,movable=False)
        self.hLine1=pg.InfiniteLine(angle=0,movable=False)
        self.hLine2=pg.InfiniteLine(angle=0,movable=False)
        self.dataPosX = 0
        
        self.label = pg.LabelItem(justify="left")
        hbox = QVBoxLayout()
		
        self.setLayout(hbox)
        self.plotwidget = pg.PlotWidget()
        self.plotwidget.addItem(self.vLine,ignoreBounds=True)
        self.plotwidget.addItem(self.hLine1,ignoreBounds=True)
        self.plotwidget.addItem(self.hLine2,ignoreBounds=True)
        self.vb = self.plotwidget.plotItem.vb
        self.plotwidget.addItem(self.label)
        hbox.addWidget(self.plotwidget)
        
        self.plotcurves = [pg.PlotCurveItem(), pg.PlotCurveItem()]
        self.plotwidget.addItem(self.plotcurves[0])
        self.plotwidget.addItem(self.plotcurves[1])
        
        self.pen = pg.mkPen(color='blue', width=3)
        self.pen2 = pg.mkPen(color='red', width=5)
        
        # Parametrii medierii
        self.sizeWindow = 5
        self.dim = int(self.sizeWindow / 2)

        self.x = np.linspace(0, 15, 200)
        self.y = -(np.sin(self.x) + np.cos(2*self.x) + 1/3 * self.x) * np.exp(-0.2 * self.x)

        # Generate the y values with noise
        noise = np.random.normal(0, 0.1, self.x.shape) # Adjust noise amplitude as desired
        self.y = self.y + noise

        start = np.array([self.y[0]] * self.dim)
        stop = np.array([self.y[-1]] * self.dim)

        self.extendedSignal = np.concatenate((start, self.y, stop))

        self.averagedSignal = np.zeros(len(self.y))
        for i in range(self.dim, len(self.extendedSignal)-self.dim-1):
            self.averagedSignal[i-self.dim] = np.mean(self.extendedSignal[i-self.dim : i+self.dim+1])
        
        self.updateplot()
        
        self.plotwidget.setRange(yRange=[min(self.y)-0.1, max(self.y)+0.1])
        
        self.curve1Points = pg.CurvePoint(self.plotcurves[0])
        self.curve2Points = pg.CurvePoint(self.plotcurves[1])
        self.plotwidget.addItem(self.curve1Points)
        self.plotwidget.addItem(self.curve2Points)
        
        self.texts = [pg.TextItem(str(1), color='blue', anchor=(0.5,-0.5)), pg.TextItem(str(1), color='red', anchor=(0.5,-0.5))]
        self.texts[0].setParentItem(self.curve1Points)
        self.texts[1].setParentItem(self.curve2Points)
        
        self.proxy = pg.SignalProxy (self.plotwidget.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        
    def mouseMoved(self, evt):
        pos = evt[0] 
		
        if self.plotwidget.sceneBoundingRect().contains(pos):
            mousePoint = self.vb.mapSceneToView(pos)
            index = int((mousePoint.x() * 200) / 15)
            
            if index >=0 and index < len(self.x):
                dataPosX = mousePoint.x()
				
                self.curve1Points.setPos(float(index)/(len(self.x)-1))
                self.texts[0].setText("%0.3f, %0.3f"%(dataPosX, self.y[index]))
                
                self.curve2Points.setPos(float(index)/(len(self.x)-1))
                self.texts[1].setText("%0.3f, %0.3f"%(dataPosX, self.averagedSignal[index]))
			
                self.vLine.setPos(mousePoint.x())
                self.hLine1.setPos(self.y[index])
                self.hLine2.setPos(self.averagedSignal[index])
    
    def updateplot(self):
        self.plotcurves[0].setData(self.x, self.y, pen=self.pen, clickable=True)
        self.plotcurves[1].setData(self.x, self.averagedSignal, pen=self.pen2, clickable=True)

    def applyNewSettings(self, sizeWindow):
        self.sizeWindow = sizeWindow
        self.dim = int(self.sizeWindow / 2)

        start = np.array([self.y[0]] * self.dim)
        stop = np.array([self.y[-1]] * self.dim)

        self.extendedSignal = np.concatenate((start, self.y, stop))

        self.averagedSignal = np.zeros(len(self.y))
        for i in range(self.dim, len(self.extendedSignal)-self.dim-1):
            self.averagedSignal[i-self.dim] = np.mean(self.extendedSignal[i-self.dim : i+self.dim+1])
        
        self.updateplot()
