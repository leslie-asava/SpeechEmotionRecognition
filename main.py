import sys
import pyqtgraph as pg
import numpy as np
import os
from PyQt5.QtWidgets import QMainWindow,QLabel, QApplication,QPushButton, QFileDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from scipy.io import wavfile as wav

import glob  
import os  
import librosa  
import numpy as np  
import keras
import pickle
from keras.models import Sequential  
from keras.layers import Dense, Activation  
from keras.layers import Dropout  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix
import pandas as pd  
import seaborn as sns  
import matplotlib.pyplot as plt

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


class FeatureExtractionThread(QThread):
	done = pyqtSignal(str)
	def run(self):
		self.done.emit("Extraction in Progress")

		main_dir = 'Audio_Speech_Actors_01-24'  
		sub_dir=os.listdir(main_dir)  
		print ("\ncollecting features and labels...")  
		print("\nthis will take some time...")  
		features, labels = parse_audio_files(main_dir,sub_dir)  
		print("done")   
		#one hot encoding labels  
		labels = one_hot_encode(labels)  

		with open('X.pickle','wb') as file:
			pickle.dump(features,file)

		with open('y.pickle','wb') as file:
			pickle.dump(labels,file)

		self.done.emit("Extraction Complete")

	
class TrainNetworkThread(QThread):
	done = pyqtSignal(str)
	def run(self):
		self.done.emit("Training in Progress")

		input_file = open('X.pickle','rb')
		X = pickle.load(input_file)

		input_file = open('y.pickle','rb')
		y = pickle.load(input_file)

		train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.33, random_state=42)

		#dnn parameters
		n_dim = train_x.shape[1]  
		n_classes = train_y.shape[1]  
		n_hidden_units_1 = n_dim  
		n_hidden_units_2 = 400 # approx n_dim * 2  
		n_hidden_units_3 = 200 # half of layer 2  
		n_hidden_units_4 = 100

		#defining the model
		def create_model(activation_function='relu', optimiser='adam', dropout_rate=0.2):
			model = Sequential()
			# layer 1
			model.add(Dense(n_hidden_units_1, input_dim=n_dim, activation=activation_function))
			# layer 2
			model.add(Dense(n_hidden_units_2, activation=activation_function))
			model.add(Dropout(dropout_rate))
			# layer 3
			model.add(Dense(n_hidden_units_3, activation=activation_function))
			model.add(Dropout(dropout_rate))
			#layer4
			model.add(Dense(n_hidden_units_4, activation=activation_function))
			model.add(Dropout(dropout_rate))
			# output layer
			model.add(Dense(n_classes, activation='softmax'))
			#model compilation
			model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
			return model

		#create the model  
		model = create_model()  
		#train the model  
		history = model.fit(train_x, train_y, epochs=200, batch_size=4)

		model.save("model")

		self.done.emit("Training Complete")


class RecognitionThread(QThread):
	done = pyqtSignal(str)
	audio_file = ""
	def run(self):
		model = keras.models.load_model("model")
		features, labels = np.empty((0,193)), np.empty(0)

		mfccs, chroma, mel, contrast,tonnetz = extract_feature(self.audio_file)
		ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
		features = np.vstack([features,ext_features])

		predict=model.predict(features,batch_size=1)
		percentage = "%.1f" % (max(predict[0]) * 100)
		print(percentage)

		emotions=['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']  
		#predicted emotions from the test set  
		y_pred = np.argmax(predict, 1)
		self.done.emit(str(emotions[int(y_pred)]) + " -> " + str(percentage) + "%")



class MainWindow(QMainWindow):

	def __init__(self):
		super().__init__()
		self.createUI()
		self.format_buttons()
		self.show()

		self.recognition_thread = RecognitionThread()
		self.recognition_thread.done.connect(self.update_emotion)
		self.feature_extraction_thread = FeatureExtractionThread()
		self.feature_extraction_thread.done.connect(self.update_emotion)
		self.train_network_thread = TrainNetworkThread()
		self.train_network_thread.done.connect(self.update_emotion)

		self.audio_file = ""

	def plot_graph(self,fileName):
		rate, data = wav.read(fileName)
		data = np.array(data, dtype=np.float32)
		green_pen = pg.mkPen(color=(0, 200, 0), width=1)
		self.graphWidget.clear()
		self.graphWidget.plot(data,pen=green_pen)

	def openAudioFileDialog(self):
		self.emotion_label.setText("")
		options = QFileDialog.Options()
		fileName, _ = QFileDialog.getOpenFileName(self,"Select audio File", "","WAV Files (*.wav);", options=options)
		if fileName:
			self.audio_file = fileName
			self.graphWidget.setTitle("<span style=\"color:white;font-size:19px\">" + str(os.path.basename(fileName)) + "</span>")
			self.plot_graph(fileName)

	def update_emotion(self,text):
		self.emotion_label.setText(text)

	def format_buttons(self):
		x = 50
		y = 130
		width = 170
		height = 60
		for button in self.button_list:
			button.resize(width,height)
			button.move(x,y)
			#button.setStyleSheet("font-size:9pt")
			y += 100

	def on_recognition_button_click(self):
		self.recognition_thread.audio_file = self.audio_file
		self.recognition_thread.start()

	def on_training_button_click(self):
		self.train_network_thread.start()

	def on_extraction_button_click(self):
		self.feature_extraction_thread.start()

	def createUI(self):
		X = 400
		Y = 150
		WIDTH = 1040
		HEIGHT = 640
		self.setGeometry(X, Y, WIDTH, HEIGHT)
		self.setWindowTitle("Speech Emotion Recognition")

		self.title_label = QLabel(self)
		self.title_label.setText("Speech Emotion Recognition using Neural Network")
		self.title_label.resize(600,50)
		self.title_label.move(370,20)
		self.title_label.setStyleSheet("font-size:17pt;font-family:calibri;color:blue;font-weight:bold")

		self.extraction_button = QPushButton(self)
		self.extraction_button.setText("Feature Extraction")
		self.extraction_button.clicked.connect(self.on_extraction_button_click)

		self.training_button = QPushButton(self)
		self.training_button.setText("Network Training")
		self.training_button.clicked.connect(self.on_training_button_click)

		self.browse_button = QPushButton(self)
		self.browse_button.setText("Browse Audio")
		self.browse_button.clicked.connect(self.openAudioFileDialog)

		self.recognition_button = QPushButton(self)
		self.recognition_button.setText("Emotion Recognition")
		self.recognition_button.clicked.connect(self.on_recognition_button_click)

		self.emotion_label = QLabel(self)
		self.emotion_label.setText("")
		self.emotion_label.resize(300,50)
		self.emotion_label.move(480,550)
		self.emotion_label.setStyleSheet("font-size:18pt;font-weight:bold;font-family:calibri;background-color:rgb(80,80,80);border-radius:10px;color:rgb(0,210,0)")
		self.emotion_label.setAlignment(Qt.AlignCenter)

		self.graphWidget = pg.PlotWidget(self)
		self.graphWidget.resize(650,420)
		self.graphWidget.move(270,100)
		self.graphWidget.setBackground("#414141")
		self.graphWidget.setTitle("<span style=\"color:white;font-size:19px\">Audio signal</span>")

		self.button_list = [self.extraction_button, self.training_button, self.browse_button, self.recognition_button]

if __name__ == '__main__':
	app = QApplication(sys.argv)
	app.setStyle('Fusion')
	window = MainWindow()
	sys.exit(app.exec_()) 
