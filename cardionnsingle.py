#Kai Kuspa and Alex Lin
#Treehacks15

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
import csv
from numpy import genfromtxt
import numpy as np
from datetime import datetime
startTime = datetime.now()

TRAININGS = 100
START_SLICE = 0
EVT_START = 72500
EVT_END = 75000

ARRHYTHMIA_TRUE = 1
ARRHYTHMIA_FALSE = 0

WINDOW_SIZE = 250
NUM_HIDDEN_STATES = 3
OUTPUT_STATES = 1

NUM_PATIENTS = 30

non_evt_iterations = 200
evt_iterations = 10

'''Declare dataset to train NN'''
ds = SupervisedDataSet(WINDOW_SIZE, OUTPUT_STATES)

'''Iterate over slice of training dataset and add parameter to SupervisedDataSet'''
def addParametersFromData(trainingData, outputParam):

	window = []
	counter = 0
	for data in trainingData:

		window.append(data)
		
		if counter < WINDOW_SIZE - 1:
			counter = counter + 1
		else:
			ds.addSample(np.array(window), (outputParam,))
			window = []
			counter = 0

'''Read in data from patients'''
patients = []
for n in range(1,NUM_PATIENTS+1):
	print("Loading patient " + str(n))
	dataname = "patient_" + str(n) 
	data = genfromtxt(dataname + '.csv', delimiter=',', skip_footer=2)
	patients.append(data)

print("Loaded " + str(len(patients)) + " patients.")


'''Train model'''

startRange = 0
#buffer = []
for i in range(0, non_evt_iterations):
	counter = 1
	for p in patients:
		startRange = WINDOW_SIZE * i
		tempSlice = p[startRange:startRange + WINDOW_SIZE]

		print("Adding negative paramters to neural net from patient " + str(counter))
		counter = counter + 1
		addParametersFromData(tempSlice, ARRHYTHMIA_FALSE)

for i in range(0, evt_iterations):
	counter = 1
	startRange = EVT_START
	for pa in patients:
		evtSlice = pa[startRange:startRange + WINDOW_SIZE]
		
		print("Adding positive paramters to neural net from patient " + str(counter))
		addParametersFromData(evtSlice, ARRHYTHMIA_TRUE)
		counter = counter + 1
		startRange + WINDOW_SIZE





'''Create NN with 250 input stream, 3 hidden values, one output'''
net = buildNetwork(WINDOW_SIZE, NUM_HIDDEN_STATES, OUTPUT_STATES, bias=True, hiddenclass=TanhLayer)
trainer = BackpropTrainer(net, ds)

'''Train for a number of cycles'''
for y in range(0,TRAININGS):
	print("training iteration " + str(y))
	trainer.train()



result = net.activate((-1369,-1429,-1427,-1172,-846,-752,-742,-552,-285,-295,-628,-916,-856,-605,-504,-654,-742,-629,-514,-570,-628,-474,-228,-111,-114,-125,-114,-26,285,650,571,-66,-635,-751,-456,-280,-285,-235,-114,-79,-114,-81,0,8,-57,-110,-171,-315,-514,-654,-685,-628,-514,-391,-342,-402,-524,-575,-514,-297,0,132,-114,-602,-913,-856,-685,-678,-742,-642,-399,-217,-171,-188,-230,-272,-228,-16,285,506,571,535,456,333,171,68,114,249,285,159,0,-89,-171,-329,-456,-395,-171,36,57,-187,-685,-1250,-1598,-1584,-1255,-585,571,2115,3558,4319,4337,3744,2682,1332,57,-799,-1198,-1309,-1255,-1116,-1027,-1067,-1141,-1113,-970,-737,-486,-301,-456,-749,-742,-421,-285,-608,-970,-939,-685,-560,-514,-353,-228,-327,-456,-339,-114,-87,-228,-262,-114,56,114,45,-124,-231,-171,-20,-57,-355,-628,-605,-399,-292,-342,-400,-399,-404,-456,-523,-571,-571,-514,-454,-456,-443,-285,-93,-171,-574,-913,-865,-571,-416,-514,-643,-628,-523,-456,-443,-399,-274,-114,-12,0,-3,57,150,171,90,0,30,171,278,228,126,171,329,342,84,-228,-368,-342,-237,-57,97,-57,-558,-1084,-1369,-1541,-1737,-1712,-1132,-57,1186,2397,3485,4222,4308,3652,2460,1084,-107,-856,-1160,-1255,-1351,-1427,-1352,-1141,-972,-970,-1037,-970,-710,-456,-466,-742,-1005,-970,-690))

if result < 0.05:
	print("Vitals normal")
else:
	print("WARNING: Arrhythmia detected!")




<<<<<<< HEAD
print('It took ', datetime.now() - startTime, 'seconds and ', ITERATIONS, 'iteration(s) to train and confirm heart model.')
=======
print 'It took ', datetime.now() - startTime, 'seconds and ', TRAININGS, 'iteration(s) to train and confirm heart model.'
>>>>>>> origin/master
