import sys
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model

MODEL_LOAD = True
MODEL_SAVE = False
DISPLAY = True

X_train_dir = sys.argv[3]
Y_train_dir = sys.argv[4]

X_test_dir = sys.argv[5]
Y_test_dir = sys.argv[6]

Model_load_dir = 'hw2_best.h5'
Model_save_dir = 'Yo.h5'

X_train = np.array(pd.read_csv(X_train_dir))
Y_train = np.array(pd.read_csv(Y_train_dir))
X_test 	= np.array(pd.read_csv(X_test_dir))

# Normalization
X_train_max = X_train.max(axis = 0)
X_train_min = X_train.min(axis = 0)
X_train = (X_train - X_train_min) / (X_train_max - X_train_min)
X_test 	= (X_test - X_train_min) / (X_train_max - X_train_min)

model = Sequential()

if(MODEL_LOAD == True):
	model = load_model(Model_load_dir)

else:
	model.add(Dense(input_dim = 106, kernel_initializer = 'normal', output_dim = 1024, activation = 'relu'))
	model.add(Dropout(0.2))
	model.add(Dense(kernel_initializer = 'normal', output_dim = 1024, activation = 'relu'))
	model.add(Dropout(0.2))
	model.add(Dense(output_dim = 1, activation = 'sigmoid'))

	model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

	model.fit(X_train, Y_train, batch_size = 100, epochs = 5)

if(MODEL_SAVE == True):
	model.save(Model_save_dir)

y = model.predict(X_test, batch_size = 100)

result = np.around(y)

if(DISPLAY == True):
	print(result)

with open(Y_test_dir, 'w') as output:
	output.write('id,label\n')
	for i in range(len(result)):
		output.write(str(i+1))
		output.write(',')
		output.write(str(int(result[i])))
		output.write('\n')














