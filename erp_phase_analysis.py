# Author: Couto, B.A.N.
# Date: Aug, 2023
# Description: This script contains an application and some methods for phase analysis of ERP data.

# %% Imports
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import hilbert

# PyQT5
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QListWidget, QWidget, QLabel, QComboBox, QLineEdit, QPushButton, QFileDialog, QShortcut
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# %% Methods:
# Hilbert Cross Spectrum
def calculate_cross_spectrum_hilbert(data, chs1, chs2):
    """
    Calculate the cross-spectrum between two channels in an mne.Epochs object using the Hilbert transform.

    Parameters:
        epochs (mne.Epochs): The epochs object containing the data.
        ch1 (int): Index of the first channel.
        ch2 (int): Index of the second channel.
        fband (tuple): Frequency band of interest (low, high). Defaults to 'None'.

    Returns:
        cross_spectrum (numpy.ndarray): Cross-spectral density.
    """
    # Extract data for the given channels
    data_chs1 = data[:, chs1, :]
    data_chs2 = data[:, chs2, :]

    # Compute the analytic signals using the Hilbert transform
    analytic_signal_ch1 = hilbert(data_chs1)
    analytic_signal_ch2 = hilbert(data_chs2)

    # Compute the cross-spectrum
    cross_spectrum = analytic_signal_ch1 * np.conj(analytic_signal_ch2)

    return cross_spectrum

# PLV: Phase Locking Value
def calculate_plv(data, ch1, ch2):
    """
    Calculate the Phase Locking Value (PLV) between two channels in an mne.Epochs object.

    Parameters:
        epochs (mne.Epochs): The epochs object containing the data.
        ch1 (str): Name of the first channel.
        ch2 (str): Name of the second channel.

    Returns:
        plv (numpy.ndarray): Phase Locking Value.
    """
    # Extract data for the given channels
    data_ch1 = data[:, ch1, :]
    data_ch2 = data[:, ch2, :]

    # Compute the analytic signals using the Hilbert transform
    analytic_signal_ch1 = hilbert(data_ch1)
    analytic_signal_ch2 = hilbert(data_ch2)

    # Compute the phase difference
    phase_difference = np.angle(analytic_signal_ch1) - np.angle(analytic_signal_ch2)

    # Normalize the phase difference
    complex_phase_difference = np.exp(1j * phase_difference)

    # Compute the mean vector across trials
    mean_vector = np.mean(complex_phase_difference, axis=0)

    # Compute the PLV
    plv = np.abs(mean_vector)

    return plv

# iPLV: Imaginary Phase Locking Value
def calculate_iplv(data, ch1, ch2):
    """
    Calculate the Imaginary Phase Locking Value (iPLV) between two channels in an mne.Epochs object.

    Parameters:
        epochs (mne.Epochs): The epochs object containing the data.
        ch1 (str): Name of the first channels.
        ch2 (str): Name of the second channels.

    Returns:
        iplv (numpy.ndarray): Phase Locking Value.
    """
    # Extract data for the given channels
    data_ch1 = data[:, ch1, :]
    data_ch2 = data[:, ch2, :]

    # Compute the analytic signals using the Hilbert transform
    analytic_signal_ch1 = hilbert(data_ch1)
    analytic_signal_ch2 = hilbert(data_ch2)

    # Compute the phase difference
    phase_difference = np.angle(analytic_signal_ch1) - np.angle(analytic_signal_ch2)

    # Normalize the phase difference
    complex_phase_difference = np.exp(1j * phase_difference)

    # Get only the imaginary part
    complex_phase_difference = complex_phase_difference.imag

    # Compute the mean vector across trials
    mean_vector = np.mean(complex_phase_difference, axis=0)

    # Compute the iPLV
    iplv = np.abs(mean_vector)

    return iplv

# PLI: Phase-Lag Index
def calculate_pli(data, ch1, ch2):
    """
    Calculate the Imaginary Phase Locking Value (iPLV) between two channels in an mne.Epochs object.

    Parameters:
        epochs (mne.Epochs): The epochs object containing the data.
        ch1 (str): Name of the first channels.
        ch2 (str): Name of the second channels.

    Returns:
        plv (numpy.ndarray): Phase Locking Value.
    """
    # Extract data for the given channels
    data_ch1 = data[:, ch1, :]
    data_ch2 = data[:, ch2, :]

    # Compute the analytic signals using the Hilbert transform
    analytic_signal_ch1 = hilbert(data_ch1)
    analytic_signal_ch2 = hilbert(data_ch2)

    # Compute the phase difference
    phase_difference = np.angle(analytic_signal_ch1) - np.angle(analytic_signal_ch2)

    # Get the sign of the phase differrence
    complex_phase_difference = np.sign(phase_difference)

    # Compute the mean vector across trials
    mean_vector = np.mean(complex_phase_difference, axis=0)

    # Compute the PLI
    pli = np.abs(mean_vector)

    return pli

# wPLI: Weighted Phase-Lag Index
def calculate_wpli(data, ch1, ch2):
    # Compute the cross-spectrum
    s = calculate_cross_spectrum_hilbert(data, ch1, ch2)

    # Compute the weighted phase-lag index
    num = np.abs(np.mean(np.abs(s.imag)*np.sign(s.imag), axis=0))
    denom = np.mean(np.abs(s.imag), axis=0)
    wpli = num / denom
    return wpli

# %% Application:
class MyApp(QMainWindow):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs.copy()
        self.filtered_epochs = self.epochs.copy()
        self.methods = {
            0: "Phase Locking Value (PLV)",
            1: "Imaginary Phase Locking Value (iPLV)",
            2: "Phase Lag Index (PLI)",
            3: "Weighted Phase-Lag Index (wPLI)"
        }
        self.channels = self.epochs.ch_names
        self.data = {}
        self.returnValue = None
        self.initUI()
        
        # Update the plot
        self.update_plot()
        
        # Show the window
        self.show()
        self.activateWindow()
        self.raise_()
        self.setWindowState(Qt.WindowActive)

    def initUI(self):
        # Set up the main window
        self.setWindowTitle("ERP Phase Analysis")
        self.setGeometry(100, 100, 800, 600)

        # Initialize the main layout
        layout = QVBoxLayout()

        # Initialize the filter layout
        filter_layout = QHBoxLayout()
        self.filterLabel = QLabel("Filter:")
        self.filterLineEdit = QLineEdit()
        self.filterLineEdit.setPlaceholderText("e.g. [8, 12]")
        self.filterButton = QPushButton("Apply")
        self.filterButton.clicked.connect(self.apply_filter)
        filter_layout.addWidget(self.filterLabel)
        filter_layout.addWidget(self.filterLineEdit)
        filter_layout.addWidget(self.filterButton)
        layout.addLayout(filter_layout)

        # Initialize the methods layout
        methods_layout = QHBoxLayout()
        self.methodLabel = QLabel("Method:")
        self.methodComboBox = QComboBox()
        self.methodComboBox.addItems(list(self.methods.values()))
        self.methodComboBox.currentIndexChanged.connect(self.update_plot)
        methods_layout.addWidget(self.methodLabel)
        methods_layout.addWidget(self.methodComboBox)
        layout.addLayout(methods_layout)

        # Create the two list widgets
        channelsLayout = QHBoxLayout()

        self.list1 = QListWidget(self)
        self.list1.addItems(self.channels)
        self.list1.setCurrentRow(0)

        self.list2 = QListWidget(self)
        self.list2.addItems(self.channels)
        self.list2.setCurrentRow(0)

        channelsLayout.addWidget(self.list1)
        channelsLayout.addWidget(self.list2)
        
        self.addChannelsButton = QPushButton("Add")
        self.addChannelsButton.clicked.connect(self.add_channels)
        self.addChannelsButton.setShortcut(QtGui.QKeySequence("Ctrl+A"))
        channelsLayout.addWidget(self.addChannelsButton)

        layout.addLayout(channelsLayout)

        # Set up the Matplotlib figure and canvas
        self.figure, self.ax = plt.subplots(layout='tight')
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Add Plot Buttons Layout
        plotButtonsLayout = QHBoxLayout()
        self.resetPlotButton = QPushButton("Reset")
        self.resetPlotButton.clicked.connect(self.update_plot)
        self.resetPlotButton.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        plotButtonsLayout.addWidget(self.resetPlotButton)
        self.savePlotButton = QPushButton("Save")
        self.savePlotButton.clicked.connect(self.save_plot)
        self.savePlotButton.setShortcut(QtGui.QKeySequence("Ctrl+S"))
        plotButtonsLayout.addWidget(self.savePlotButton)
        layout.addLayout(plotButtonsLayout)


        # Set the layout to the central widget of the main window
        central_widget = QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Set up the shortcut for closing the app
        close_shortcut = QShortcut(QtGui.QKeySequence("Ctrl+Q"), self)
        close_shortcut.activated.connect(self.close_app)

    def apply_filter(self):
        self.filterButton.label = "Applying..."
        filter = self.filterLineEdit.text()
        if filter == "":
            self.filtered_epochs = self.epochs.copy()
            self.filterButton.label = "Apply"
        else:
            filter = filter.replace("[", "")
            filter = filter.replace("]", "")
            filter = filter.split(",")
            filter = [int(x) for x in filter]
            self.filtered_epochs = self.epochs.copy()
            self.filtered_epochs = self.filtered_epochs.filter(filter[0], filter[1], method='iir')
            self.filterButton.label = "Apply"
        self.update_plot()

    def add_channels(self):
        # Get which method is selected
        method = self.methodComboBox.currentIndex()

        # Get which channels are selected
        channel1 = self.list1.currentItem().text()
        channel1_idx = self.filtered_epochs.ch_names.index(channel1)
        channel2 = self.list2.currentItem().text()
        channel2_idx = self.filtered_epochs.ch_names.index(channel2)
        
        # Run Analysis
        data = self.run_phase_analysis(method, self.filtered_epochs.get_data(), channel1_idx, channel2_idx)
        self.data[f"{channel1} - {channel2}"] = data

        # Plot
        self.ax.plot(self.filtered_epochs.times, data, label=f"{channel1} - {channel2}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Phase Analysis")
        self.ax.set_xlim(self.filtered_epochs.times[0], self.filtered_epochs.times[-1])
        self.ax.legend()
        self.canvas.draw()

    def run_phase_analysis(self, method, epochs, channel1, channel2):
        if method == 0:
            return calculate_plv(epochs, channel1, channel2)
        elif method == 1:
            return calculate_iplv(epochs, channel1, channel2)
        elif method == 2:
            return calculate_pli(epochs, channel1, channel2)
        elif method == 3:
            return calculate_wpli(epochs, channel1, channel2)
        else:
            return None

    def update_plot(self):
        # Get which method is selected
        method = self.methodComboBox.currentIndex()

        # Check it self.ax is defined, else do nothing
        # Clear the current axes
        self.data = {}
        self.ax.clear()

        # Here, you can implement the logic to change the plot based on the selected items
        # For this example, we will plot random data
        #data = np.random.randn(10)

        self.ax.set_title(f"{self.methods[method]}")

        # Redraw the canvas
        self.canvas.draw()

    def save_plot(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG Files (*.png);;JPEG Files (*.jpg);; SVG Files (*.svg);;All Files (*)", options=options)
        print(fileName)
        if fileName:
            self.figure.savefig(fileName)

    def close_app(self):
        self.close()

    def closeEvent(self, event):
        self.figure.clf()
        plt.close('all')
        self.returnValue = self.data
        event.accept()

# %% Main Function:
qt_app = None
def ERPPhaseApp(epochs):
    global qt_app

    if qt_app is None:
        qt_app = QApplication(sys.argv)

    ex = MyApp(epochs)
    ex.show()

    try:
        qt_app.exec_()
    except SystemExit:
        pass
    return ex.returnValue