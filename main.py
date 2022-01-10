import sys
import random
from PyQt5.QtWidgets import QMainWindow,QLabel,QApplication,QPushButton,QFileDialog,QProgressBar,QLineEdit,QMessageBox,QWidget,QGraphicsOpacityEffect,QComboBox, QFormLayout
from PyQt5.QtGui import QPixmap,QFont,QMovie,QPainter,QBrush,QPen
from PyQt5.QtCore import Qt, QThread,pyqtSignal,QByteArray
from scipy.io import wavfile as wav
from PyQt5 import QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import pickle
import os
import pyqtgraph as pg
import numpy as np
import keras
import librosa

X = 400
Y = 150
WIDTH = 1200
HEIGHT = 800

def extract_feature(file_name): 
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
              print ("Error encountered while parsing file: ", fn)
              continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            labels = np.append(labels, fn.split('\\')[2].split('-')[2])
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels+1))
    one_hot_encode[np.arange(n_labels), labels] = 1
    one_hot_encode=np.delete(one_hot_encode, 0, axis=1)
    return one_hot_encode

class RecognitionThread(QThread):
	accuracy = pyqtSignal(str)
	filename = pyqtSignal(str)
	done = pyqtSignal()
	def run(self):
		model = keras.models.load_model("model")
		features, labels = np.empty((0,193)), np.empty(0)

		for audio_file in controller.file_list:

			mfccs, chroma, mel, contrast,tonnetz = extract_feature(audio_file)
			ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
			features = np.vstack([features,ext_features])

			predict=model.predict(features,batch_size=1)
			percentage = "%.1f" % (max(predict[0]) * 100)

			emotions=['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']  
			#predicted emotions from the test set  
			y_pred = np.argmax(predict, 1)
			self.filename.emit(audio_file)
			self.accuracy.emit(str(emotions[int(y_pred[-1])]) + " > " + str(percentage) + "%")
			if str(emotions[int(y_pred[-1])]).lower() == "neutral":
				controller.neutral_list.append(audio_file)
			elif str(emotions[int(y_pred[-1])]).lower() == "calm":
				controller.calm_list.append(audio_file)
			elif str(emotions[int(y_pred[-1])]).lower() == "happy":
				controller.happy_list.append(audio_file)
			elif str(emotions[int(y_pred[-1])]).lower() == "sad":
				controller.sad_list.append(audio_file)
			elif str(emotions[int(y_pred[-1])]).lower() == "angry":
				controller.angry_list.append(audio_file)
			elif str(emotions[int(y_pred[-1])]).lower() == "fearful":
				controller.fearful_list.append(audio_file)
			elif str(emotions[int(y_pred[-1])]).lower() == "disgust":
				controller.disgust_list.append(audio_file)
			elif str(emotions[int(y_pred[-1])]).lower() == "surprised":
				controller.surprised_list.append(audio_file)
		self.done.emit()

class ChartCanvas(FigureCanvas):
    def __init__(self, parent):
        fig, self.ax = plt.subplots(figsize=(10.8, 2.8), dpi = 75)
        #fig.set_facecolor("#B1B1B1")
        super().__init__(fig)
        self.setParent(parent)

class WidgetGroup(QWidget):
	def __init__(self, parent,x,y,title,color):
		super().__init__()
		self.parent = parent
		self.color = color
		self.title = title
		self.width, self.height = 260,320
		self.x,self.y = x,y
		self.createUI()
		self.panel.move(x, y)
		self.layout = QFormLayout(self.panel)

	def createUI(self):
		self.panel = QLabel(self.parent)
		self.panel.resize(self.width,self.height)
		self.panel.setStyleSheet("background-color:white")

		self.title_label = QLabel(self.parent)
		self.title_label.resize(self.width,40)
		self.title_label.setText(self.title)
		self.title_label.setStyleSheet("background-color:%s; padding-left:10px" %self.color)
		self.title_label.move(self.x,self.y-35)

		

class WindowTemplate():
	def __init__(self,parent):
		super().__init__()
		self.parent = parent
		self.createUI()

	def on_home_btn_click(self):
		controller.current_screen = "home"
		self.parent.switch_to_home.emit()
		self.parent.close()

	def on_browse_audio_file_btn_click(self):
		controller.current_screen = "file"
		self.parent.switch_to_file.emit()
		self.parent.close()

	def on_browse_audio_folder_btn_click(self):
		controller.current_screen = "folder"
		self.parent.switch_to_folder.emit()
		self.parent.close()

	def on_emotion_recognition_btn_click(self):
		controller.current_screen = "emotion"
		self.parent.switch_to_emotion.emit()
		self.parent.close()

	def on_analyze_data_btn_click(self):
		controller.current_screen = "analysis"
		self.parent.switch_to_analysis.emit()
		self.parent.close()

	def on_desired_btn_click(self):
		controller.current_screen = "desired"
		self.parent.switch_to_desired.emit()
		self.parent.close()

	def update_active_screen(self):
		self.format_side_bar()
		button = self.home_btn
		if controller.current_screen == "home":
			button = self.home_btn
		elif controller.current_screen == "file":
			button = self.browse_audio_file_btn
		elif controller.current_screen == "folder":
			button = self.browse_audio_folder_btn
		elif controller.current_screen == "emotion":
			button = self.emotion_recognition_btn
		elif controller.current_screen == "analysis":
			button = self.analyze_data_btn
		elif controller.current_screen == "desired":
			button = self.desired_btn
		button.setStyleSheet("QPushButton"
                             "{"
                             "background-color : #484E55;color:#FFFEF6;border: solid #4B80F5;border-width: 0px 0px 0px 7px;font-size:8.7pt;"
                             "}")

	def format_side_bar(self):
		start_x = 0
		start_y = 120
		button_width = 227
		button_height = 53
		for button in self.button_list:
			button.resize(button_width,button_height)
			button.move(start_x,start_y)
			button.setStyleSheet("QPushButton::hover"
                             "{"
                             "background-color : #00617F;color:#FFFEF6;border:None"
                             "}"
                             "QPushButton"
                             "{"
                             "font-size:8.7pt;background-color:#252525;color:#FFFEF6; border : None"
                             "}"
                             )
			start_y+=53

	def createUI(self):
		self.parent.setWindowTitle("Speech Emotion Recognition")
		self.footer_opacity_effect = QGraphicsOpacityEffect()
		self.footer_opacity_effect.setOpacity(0.8)
		self.panel_opacity_effect = QGraphicsOpacityEffect()
		self.panel_opacity_effect.setOpacity(0.8)

		self.background = QLabel(self.parent)
		self.background.resize(WIDTH, HEIGHT)
		self.background.setStyleSheet("background-color:#333333")
		
		self.buttons_panel = QLabel(self.parent)
		self.buttons_panel.move(0,80)
		self.buttons_panel.resize(237,HEIGHT-87)
		self.buttons_panel.setStyleSheet("background-color : #252525")
		#self.buttons_panel.setGraphicsEffect(self.panel_opacity_effect)

		self.widget_panel = QLabel(self.parent)
		self.widget_panel.move(227,80)
		self.widget_panel.resize(965,HEIGHT-87)
		self.widget_panel.setStyleSheet("background-color : #484E55")
		#self.widget_panel.setGraphicsEffect(self.panel_opacity_effect)


		self.home_btn = QPushButton(self.parent)
		self.home_btn.setText("Home")
		self.home_btn.clicked.connect(self.on_home_btn_click)

		self.browse_audio_file_btn = QPushButton(self.parent)
		self.browse_audio_file_btn.setText("Browse Audio File")
		self.browse_audio_file_btn.clicked.connect(self.on_browse_audio_file_btn_click)

		self.browse_audio_folder_btn = QPushButton(self.parent)
		self.browse_audio_folder_btn.setText("Browse Audio Folder")
		self.browse_audio_folder_btn.clicked.connect(self.on_browse_audio_folder_btn_click)

		self.emotion_recognition_btn = QPushButton(self.parent)
		self.emotion_recognition_btn.setText("Emotion Recognition")
		self.emotion_recognition_btn.clicked.connect(self.on_emotion_recognition_btn_click)

		self.analyze_data_btn = QPushButton(self.parent)
		self.analyze_data_btn.setText("Analyze Data")
		self.analyze_data_btn.clicked.connect(self.on_analyze_data_btn_click)

		"""self.footer_panel = QLabel(self.parent)
		self.footer_panel.move(0,HEIGHT-50)
		self.footer_panel.resize(WIDTH,50)
		self.footer_panel.setStyleSheet("background-color : rgb(20,20,20)")
		self.footer_panel.setGraphicsEffect(self.footer_opacity_effect)"""
		self.python=QLabel(self.parent)
		self.python_pixmap=QPixmap('Python-Logo-Png.png')
		#python_pixmap = python_pixmap.scaledToHeight(35)
		self.python.setPixmap(self.python_pixmap)
		self.python.resize(230,60)
		self.python.move(640,HEIGHT-55)

		self.button_list = [self.home_btn,self.browse_audio_file_btn,self.browse_audio_folder_btn,self.emotion_recognition_btn,self.analyze_data_btn]
		self.update_active_screen()
		


class HomeScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_file = pyqtSignal()
	switch_to_folder = pyqtSignal()
	switch_to_emotion = pyqtSignal()
	switch_to_analysis = pyqtSignal()	
	switch_to_desired = pyqtSignal()

	def __init__(self):
		super().__init__()
		self.createUI()

	def createUI(self):
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.template = WindowTemplate(self)

		self.logo_label = QLabel(self)
		self.logo_label.setPixmap(QtGui.QPixmap("sound_logo.png").scaledToWidth(500))
		self.logo_label.resize(500,240)
		self.logo_label.move(430,100)

		self.title_label = QLabel(self)
		self.title_label.setText("Speech Emotion Recognition")
		self.title_label.setStyleSheet("color:white; font-size:28pt; font-weight: bold")
		self.title_label.move(410,400)

		self.slogan_label = QLabel(self)
		self.slogan_label.setText(" ~ Know what your customers think ~ ")
		self.slogan_label.setStyleSheet("color:white; font-size:15pt")
		self.slogan_label.move(490,480)
		

class AudioFileScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_file = pyqtSignal()
	switch_to_folder = pyqtSignal()
	switch_to_emotion = pyqtSignal()
	switch_to_analysis = pyqtSignal()	
	switch_to_desired = pyqtSignal()

	def __init__(self):
		super().__init__()
		self.createUI()

		"""if controller.file_list:
			fileName = controller.file_list[0]
			self.file_path_entry.setText(fileName)
			self.graphWidget.setTitle("<span style=\"color:white;font-size:19px\">" + str(os.path.basename(fileName)) + "</span>")
			self.plot_graph(fileName)"""

	def plot_graph(self,fileName):
		rate, data = wav.read(fileName)
		data = np.array(data, dtype=np.float32)
		green_pen = pg.mkPen(color=(0, 200, 0), width=1)
		self.graphWidget.clear()
		self.graphWidget.plot(data,pen=green_pen)

	def open_audio_file_dialog(self):
		#self.emotion_label.setText("")
		options = QFileDialog.Options()
		fileName, _ = QFileDialog.getOpenFileName(self,"Select audio File", "","WAV Files (*.wav);", options=options)
		if fileName:
			#self.audio_file = fileName
			controller.file_list = []

			controller.neutral_list = []
			controller.calm_list = []
			controller.happy_list = []
			controller.sad_list = []
			controller.angry_list = []
			controller.fearful_list = []
			controller.disgust_list = []
			controller.surprised_list = []

			controller.file_list.append(fileName)
			self.file_path_entry.setText(fileName)
			self.graphWidget.setTitle("<span style=\"color:white;font-size:19px\">" + str(os.path.basename(fileName)) + "</span>")
			self.plot_graph(fileName)


	def createUI(self):
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.template = WindowTemplate(self)

		self.title_label = QLabel(self)
		self.title_label.setText("Choose audio file")
		self.title_label.setStyleSheet("color:white; font-size:13pt")
		self.title_label.move(280,120)

		self.file_path_entry = QLineEdit(self)
		self.file_path_entry.resize(530,45)
		self.file_path_entry.move(280,170)
		self.file_path_entry.setStyleSheet("border-radius:10px; font-size: 10pt")

		self.file_path_btn = QPushButton(self)
		self.file_path_btn.resize(120,40)
		self.file_path_btn.move(860,172)
		self.file_path_btn.setText("Browse")
		self.file_path_btn.setStyleSheet("border-radius:10px; background-color:darkgray")
		self.file_path_btn.clicked.connect(self.open_audio_file_dialog)

		self.graphWidget = pg.PlotWidget(self)
		self.graphWidget.resize(850,450)
		self.graphWidget.move(290,270)
		self.graphWidget.setBackground("#414141")
		self.graphWidget.setTitle("<span style=\"color:white;font-size:19px\">Audio signal</span>")

class AudioFolderScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_file = pyqtSignal()
	switch_to_folder = pyqtSignal()
	switch_to_emotion = pyqtSignal()
	switch_to_analysis = pyqtSignal()	

	def __init__(self):
		super().__init__()
		self.createUI()
		self.file_index = 0

		"""if controller.file_list:
			fileName = controller.file_list[0]
			self.file_path_entry.setText(fileName)
			self.graphWidget.setTitle("<span style=\"color:white;font-size:19px\">" + str(os.path.basename(fileName)) + "</span>")
			self.plot_graph(fileName)"""

	def plot_graph(self,fileName):
		rate, data = wav.read(fileName)
		data = np.array(data, dtype=np.float32)
		green_pen = pg.mkPen(color=(0, 200, 0), width=1)
		self.graphWidget.clear()
		self.graphWidget.plot(data,pen=green_pen)

	def open_audio_folder_dialog(self):
		folder = QFileDialog.getExistingDirectory(self, "Select directory containing the HEX files")
		if folder:
			self.file_index = 0
			self.file_path_entry.setText(folder)
			controller.file_list = []

			controller.neutral_list = []
			controller.calm_list = []
			controller.happy_list = []
			controller.sad_list = []
			controller.angry_list = []
			controller.fearful_list = []
			controller.disgust_list = []
			controller.surprised_list = []

			all_files = os.listdir(folder)
			for file in all_files:
				fileName = os.path.join(folder,file)
				if file.endswith(".wav"):
					controller.file_list.append(fileName)

			self.graphWidget.setTitle("<span style=\"color:white;font-size:19px\">" + str(os.path.basename(controller.file_list[self.file_index])) + "</span>")
			self.plot_graph(controller.file_list[self.file_index])

	def on_next_button_click(self):
		if controller.file_list:
			if self.file_index < len(controller.file_list)-1:
				self.file_index += 1
			else:
				self.file_index = 0
			self.graphWidget.setTitle("<span style=\"color:white;font-size:19px\">" + str(os.path.basename(controller.file_list[self.file_index])) + "</span>")
			self.plot_graph(controller.file_list[self.file_index])

	def on_previous_button_click(self):
		if controller.file_list:
			if self.file_index > 0:
				self.file_index -= 1
			else:
				self.file_index = len(controller.file_list) - 1
			self.graphWidget.setTitle("<span style=\"color:white;font-size:19px\">" + str(os.path.basename(controller.file_list[self.file_index])) + "</span>")
			self.plot_graph(controller.file_list[self.file_index])


	def createUI(self):
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.template = WindowTemplate(self)

		self.title_label = QLabel(self)
		self.title_label.setText("Choose audio folder")
		self.title_label.setStyleSheet("color:white; font-size:13pt")
		self.title_label.move(280,120)

		self.file_path_entry = QLineEdit(self)
		self.file_path_entry.resize(530,45)
		self.file_path_entry.move(280,170)
		self.file_path_entry.setStyleSheet("border-radius:10px; font-size: 10pt")

		self.file_path_btn = QPushButton(self)
		self.file_path_btn.resize(120,40)
		self.file_path_btn.move(860,172)
		self.file_path_btn.setText("Browse")
		self.file_path_btn.setStyleSheet("border-radius:10px; background-color:darkgray")
		self.file_path_btn.clicked.connect(self.open_audio_folder_dialog)

		self.graphWidget = pg.PlotWidget(self)
		self.graphWidget.resize(850,450)
		self.graphWidget.move(290,260)
		self.graphWidget.setBackground("#414141")
		self.graphWidget.setTitle("<span style=\"color:white;font-size:19px\">Audio signal</span>")

		self.previous_button = QPushButton(self)
		self.previous_button.setText("Previous")
		self.previous_button.resize(130,40)
		self.previous_button.move(580,740)
		self.previous_button.setStyleSheet("background-color: #22577E; color: white; border-radius: 15px")
		self.previous_button.clicked.connect(self.on_previous_button_click)

		self.next_button = QPushButton(self)
		self.next_button.setText("Next")
		self.next_button.resize(130,40)
		self.next_button.move(730,740)
		self.next_button.setStyleSheet("background-color: #22577E; color: white; border-radius: 15px")
		self.next_button.clicked.connect(self.on_next_button_click)

class EmotionRecognitionScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_file = pyqtSignal()
	switch_to_folder = pyqtSignal()
	switch_to_emotion = pyqtSignal()
	switch_to_analysis = pyqtSignal()	

	def __init__(self):
		super().__init__()
		self.createUI()

		self.recognition_thread = RecognitionThread()
		self.recognition_thread.accuracy.connect(self.update_widgets)
		self.recognition_thread.filename.connect(self.plot_graph)
		self.recognition_thread.done.connect(self.done)

	def done(self):
		messagebox = QMessageBox(self)
		messagebox.setIcon(QMessageBox.Information)
		messagebox.setText("Emotion recognition successfull. You may look at the analysis      ")
		messagebox.setWindowTitle("Success")
		messagebox.show()

	def plot_graph(self,fileName):
		rate, data = wav.read(fileName)
		data = np.array(data, dtype=np.float32)
		green_pen = pg.mkPen(color=(0, 200, 0), width=1)
		self.graphWidget.clear()
		self.graphWidget.setTitle("<span style=\"color:white;font-size:19px\">" + str(os.path.basename(fileName)) + "</span>")
		self.graphWidget.plot(data,pen=green_pen)

	def update_widgets(self,text):
		self.result_label.setText(text)

	def run_model(self):
		self.recognition_thread.start()

	def createUI(self):
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.template = WindowTemplate(self)

		self.title_label = QLabel(self)
		self.title_label.setText("Emotion Recognition")
		self.title_label.setStyleSheet("color:white; font-size:13pt")
		self.title_label.move(290,120)

		self.graphWidget = pg.PlotWidget(self)
		self.graphWidget.resize(850,450)
		self.graphWidget.move(290,190)
		self.graphWidget.setBackground("#414141")
		self.graphWidget.setTitle("<span style=\"color:white;font-size:19px\">Audio signal</span>")

		self.result_label = QLabel(self)
		self.result_label.resize(300,50)
		self.result_label.setStyleSheet("color:lime; background-color: #313131; border-radius: 10px; font-size: 12pt")
		self.result_label.move(560,663)
		self.result_label.setAlignment(Qt.AlignCenter)

		self.run_emotion_recognition_button = QPushButton(self)
		self.run_emotion_recognition_button.setText("Run Model")
		self.run_emotion_recognition_button.resize(130,40)
		self.run_emotion_recognition_button.move(650,732)
		self.run_emotion_recognition_button.setStyleSheet("background-color: #22577E; color: white; border-radius: 15px")
		self.run_emotion_recognition_button.clicked.connect(self.run_model)



class DataAnalysisScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_file = pyqtSignal()
	switch_to_folder = pyqtSignal()
	switch_to_emotion = pyqtSignal()
	switch_to_analysis = pyqtSignal()	
	switch_to_desired = pyqtSignal()

	def __init__(self):
		super().__init__()
		self.createUI()
		if controller.file_list:
			self.plot_chart()
			self.display_statistics()

	def plot_chart(self):
		self.pie_chart.ax.clear()
		x = [len(controller.neutral_list),len(controller.calm_list), len(controller.happy_list), len(controller.sad_list), len(controller.angry_list), len(controller.fearful_list), len(controller.disgust_list), len(controller.fearful_list)]
		y = controller.emotion_labels

		x_new = []
		y_new = []
		index = 0
		for i in x:
			if i:
				x_new.append(i)
				y_new.append(y[index])

			index += 1

		self.pie_chart.ax.pie(x_new,labels = y_new)
		print(x,y)

	def display_statistics(self):
		files_analyzed = len(controller.file_list)
		#max_value = max(controller.file_list)
		#max_index = controller.file_list.index(max_value)

		neutral = len(controller.neutral_list)
		calm = len(controller.calm_list)
		happy = len(controller.happy_list)
		sad = len(controller.sad_list)
		angry = len(controller.angry_list)
		fearful = len(controller.fearful_list)
		disgust = len(controller.disgust_list)
		surprised = len(controller.surprised_list)

		message = ""

		message += "STATISTICS\n\nFiles Analyzed : %s\n"%files_analyzed
		message += "Neutral    : [%s/%s]   %s%s\n"%(neutral, files_analyzed, int((neutral/files_analyzed) * 100),"%")
		message += "Calm       : [%s/%s]   %s%s\n"%(calm, files_analyzed, int((calm/files_analyzed) * 100),"%")
		message += "Happy     : [%s/%s]   %s%s\n"%(happy, files_analyzed, int((happy/files_analyzed) * 100),"%")
		message += "Sad         : [%s/%s]   %s%s\n"%(sad, files_analyzed, int((sad/files_analyzed) * 100),"%")
		message += "Angry      : [%s/%s]   %s%s\n"%(angry, files_analyzed, int((angry/files_analyzed) * 100),"%")
		message += "Fearful     : [%s/%s]   %s%s\n"%(fearful, files_analyzed, int((fearful/files_analyzed) * 100),"%")
		message += "Disgust    : [%s/%s]   %s%s\n"%(disgust, files_analyzed, int((disgust/files_analyzed) * 100),"%")
		message += "Surprised : [%s/%s]   %s%s\n"%(surprised, files_analyzed, int((surprised/files_analyzed) * 100),"%")

		self.statistics_label.setText(message)

	def createUI(self):
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.template = WindowTemplate(self)

		self.title_label = QLabel(self)
		self.title_label.setText("Data Analysis")
		self.title_label.setStyleSheet("color:white; font-size:13pt")
		self.title_label.move(290,120)

		# Instantiate the pie chart canvas
		self.pie_chart = ChartCanvas(self)

		# Position the chart
		self.pie_chart.move(290,190)
		self.pie_chart.resize(620,500)

		self.statistics_label = QLabel(self)
		self.statistics_label.resize(230,500)
		self.statistics_label.move(930,190)
		self.statistics_label.setStyleSheet("background-color: #313131; border-radius: 10px; color: white; padding-left: 10px; font-size: 12pt; margin-top: 20px")

class Controller():
	def __init__(self):
		self.current_screen = ""
		self.file_list = []
		self.emotion_labels = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
		self.neutral_list = []
		self.calm_list = []
		self.happy_list = []
		self.sad_list = []
		self.angry_list = []
		self.fearful_list = []
		self.disgust_list = []
		self.surprised_list = []

	def show_home(self):
		self.home_screen = HomeScreen()
		self.home_screen.switch_to_home.connect(self.show_home)
		self.home_screen.switch_to_file.connect(self.show_file)
		self.home_screen.switch_to_folder.connect(self.show_folder)
		self.home_screen.switch_to_emotion.connect(self.show_emotion)
		self.home_screen.switch_to_analysis.connect(self.show_analysis)	
		self.home_screen.show()

	def show_file(self):
		self.file_screen = AudioFileScreen()
		self.file_screen.switch_to_home.connect(self.show_home)
		self.file_screen.switch_to_file.connect(self.show_file)
		self.file_screen.switch_to_folder.connect(self.show_folder)
		self.file_screen.switch_to_emotion.connect(self.show_emotion)
		self.file_screen.switch_to_analysis.connect(self.show_analysis)	
		self.file_screen.show()

	def show_folder(self):
		self.folder_screen = AudioFolderScreen()
		self.folder_screen.switch_to_home.connect(self.show_home)
		self.folder_screen.switch_to_file.connect(self.show_file)
		self.folder_screen.switch_to_folder.connect(self.show_folder)
		self.folder_screen.switch_to_emotion.connect(self.show_emotion)
		self.folder_screen.switch_to_analysis.connect(self.show_analysis)	
		self.folder_screen.show()

	def show_emotion(self):
		self.emotion_screen = EmotionRecognitionScreen()
		self.emotion_screen.switch_to_home.connect(self.show_home)
		self.emotion_screen.switch_to_file.connect(self.show_file)
		self.emotion_screen.switch_to_folder.connect(self.show_folder)
		self.emotion_screen.switch_to_emotion.connect(self.show_emotion)
		self.emotion_screen.switch_to_analysis.connect(self.show_analysis)	
		self.emotion_screen.show()

	def show_analysis(self):
		self.analysis_screen = DataAnalysisScreen()
		self.analysis_screen.switch_to_home.connect(self.show_home)
		self.analysis_screen.switch_to_file.connect(self.show_file)
		self.analysis_screen.switch_to_folder.connect(self.show_folder)
		self.analysis_screen.switch_to_emotion.connect(self.show_emotion)
		self.analysis_screen.switch_to_analysis.connect(self.show_analysis)	
		self.analysis_screen.show()


if __name__ == '__main__':
	app = QApplication(sys.argv)
	app.setStyle('Fusion')
	controller = Controller()
	controller.show_home()
	sys.exit(app.exec_())