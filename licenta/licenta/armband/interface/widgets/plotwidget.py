from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt
import pyqtgraph as pg

    
class PlotWidget(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)              # Initializare widget
        pg.setConfigOption("background","w")        # Fundal alb
        
        # Crearea zonei unde se pot afisa grafice
        self.canvas = pg.PlotWidget()
        self.canvas.setMouseEnabled(x=False, y=False)
        
        # Creare unui obiect de tip grafic
        self.plotcurve = pg.PlotCurveItem()
        
        # Adaugarea graficului in acea zona
        self.canvas.addItem(self.plotcurve)
        
        # Creioane
        self.pen = pg.mkPen(color=(255, 127, 14), width=3)
        self.redPen = pg.mkPen('r', width=3)
        self.greenPen = pg.mkPen('g', width=3)
        self.bluePen = pg.mkPen('b', style=Qt.DashLine, width=3)
        
        # Setari
        self.canvas.setRange(yRange=[-127, 128])    # Setarea limitelor pe axa Oy
        #self.canvas.showGrid(x=True, y=True)       # Afisarea gridului
        self.canvas.setLabel('top', " ")
        self.canvas.setLabel('bottom', "Timp")      # Denumirea axei Ox
        self.canvas.setLabel('right', " ")
        
        # Crearea unui layout vertical
        vertical_layout = QVBoxLayout()
        vertical_layout.addWidget(self.canvas)
        
        # Aplicarea layout-ului
        self.setLayout(vertical_layout)
