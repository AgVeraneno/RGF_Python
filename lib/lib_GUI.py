'''
File Name: lib_gui_v2
Version: 2017.1.17
Author: Alex Lin
'''
## exterior library
import sys
sys.path.append('../lib')
from PyQt4 import QtGui
import matplotlib.pyplot as mplot
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
## interior library
import lib_function as lf

class window(QtGui.QMainWindow):
    def __init__(self, title='untitled', x=800, y=600):
        super(window, self).__init__()
        self.__obj__()                      # initialize windows variable
        self.lyt_win = QtGui.QWidget()      # create windows widget
        self.resize(x, y)                   # change windwos size to (x,y)
        self.setWindowTitle(title)          # name windows title
    def __obj__(self):
        '''
        menu widget
        '''
        self.menu = self.menuBar()
        self.menu_act = {}                  # top menu list
        self.menu_act_list = {}             # menu action list for each top menu
        '''
        tool bar widget (not ready)
        '''
        self.toolbar = None
        self.toolbar_units = []
        '''
        dock widget (not ready)
        '''
        self.dock_widget = None
        self.status_bar = self.statusBar()
        self.central_widger = None
    def addMenuBar(self, menu_lbl='untitled', act=['untitled']):
        self.menu_act[menu_lbl] = self.menu.addMenu(menu_lbl)       # add new top menu
        thisAct = [QtGui.QAction(action, self) for action in act]   # create actions
        self.menu_act[menu_lbl].addActions(thisAct)                 # add created actions to window
        self.menu_act_list[menu_lbl] = thisAct                      # save each action to this object
    def setMenuShortCut(self, menu_lbl, SC_list=[]):
        for short_cut_idx, short_cut in enumerate(SC_list):
            self.menu_act_list[menu_lbl][short_cut_idx].setShortcut(short_cut)
    def setWindow(self, lyt):
        self.lyt_win.setLayout(lyt)
        self.setCentralWidget(self.lyt_win)
class dialog(QtGui.QWidget):
    def __init__(self, ini_path='../'):
        super(dialog, self).__init__()
        self.output_ptr = None
        self.file_path = ini_path
    def inputDialog(self, lbl_txt='', ipt_txt=''):
        obj = QtGui.QInputDialog.getText(self, lbl_txt, ipt_txt)
        return str(obj[0])
    def fileDialog(self, lbl_txt='', dir_path=None, mode='open'):
        if dir_path == None:
            dir_path = self.file_path
        else:
            pass
        if mode == 'open':
            obj = str(QtGui.QFileDialog.getOpenFileName(self, lbl_txt, dir_path))
        elif mode == 'save':
            obj = str(QtGui.QFileDialog.getSaveFileName(self, lbl_txt, dir_path))
        elif mode == 'open_m':
            obj = list(QtGui.QFileDialog.getOpenFileNames(self, lbl_txt, dir_path))
        return obj
class layout(QtGui.QWidget):
    def __init__(self, layout_type='V'):
        super(layout, self).__init__()
        self.lyt_win = QtGui.QWidget()
    def newLayout(self, layout_type=''):
        if layout_type == 'V':          # vertical box
            return QtGui.QVBoxLayout()
        elif layout_type == 'H':        # horizontal box
            return QtGui.QHBoxLayout()
        elif layout_type == 'F':        # form layout
            return QtGui.QFormLayout()
        elif layout_type == 'G':        # gric layout
            return QtGui.QGridLayout()
        else:
            raise NameError('No match layout input: '+layout_type)
    def newTab(self, tab_lbl_list=[]):
        new_tab = QtGui.QTabWidget(self)
        tab_dict = {}
        for tab_lbl in tab_lbl_list:
            new_tab_widget = QtGui.QWidget()
            new_tab.addTab(new_tab_widget, tab_lbl)
            tab_dict[tab_lbl] = new_tab_widget
        return new_tab, tab_dict
class widget(QtGui.QWidget):
    def __init__(self):
        super(widget, self).__init__()
    def newLabel(self, label_type='L', title='', spinBoxSet={}):
        if label_type == 'L':       # text label
            return QtGui.QLabel(title)
        elif label_type == 'E':     # edit label
            return QtGui.QLineEdit(title)
        elif label_type == 'PE':    # edit text box
            return QtGui.QPlainTextEdit(title)
        elif label_type == 'I':     # interger spin box
            if spinBoxSet == {}:
                return QtGui.QSpinBox()
            else:
                new_widget = QtGui.QSpinBox()
                new_widget.setRange(spinBoxSet['min'], spinBoxSet['max'])
                new_widget.setSingleStep(spinBoxSet['step'])
                new_widget.setValue(spinBoxSet['val'])
                return new_widget
        elif label_type == 'D':     # double spin box
            if spinBoxSet == {}:
                return QtGui.QDoubleSpinBox()
            else:
                new_widget = QtGui.QDoubleSpinBox()
                new_widget.setRange(spinBoxSet['min'], spinBoxSet['max'])
                new_widget.setSingleStep(spinBoxSet['step'])
                new_widget.setValue(spinBoxSet['val'])
                return new_widget
    def newButton(self, button_type='P', title=''):
        if button_type == 'P':      # push button
            return QtGui.QPushButton(title)
    def newGroup(self, group_type='C', title=''):
        if group_type == 'C':       # combobox
            new_widget = QtGui.QComboBox()
            new_widget.addItems(title)
            return new_widget
        elif group_type == 'G':       # groupbox
            return QtGui.QGroupBox(title)
    def newTable(self, row_count=1, col_count=1):
        return QtGui.QTableWidget(row_count, col_count)
        
'''
test main function
'''    
if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    thisW = window()
    thisW.show()
    sys.exit(app.exec_())