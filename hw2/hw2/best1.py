import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model

x_data_list = np.array(pd.read_csv('X_train'))
y_data_list = np.array(pd.read_csv('Y_train'))
z_data_list = np.array(pd.read_csv('X_test'))

x_data_nor_list = [[0 for col in range(len(x_data_list[0]))] for row in range(len(x_data_list))]
z_data_nor_list = [[0 for col in range(len(z_data_list[0]))] for row in range(len(z_data_list))]

for col in range(len(x_data_list[0])):
	max = 0
	for row in range(len(x_data_list)):
		if(row == 0 or x_data_list[row][col] > max):
			max = x_data_list[row][col]
	for row in range(len(x_data_list)):
		x_data_nor_list[row][col] = float(x_data_list[row][col]) / float(max+1)

for col in range(len(z_data_list[0])):
	max = 0
	for row in range(len(z_data_list)):
		if(row == 0 or z_data_list[row][col] > max):
			max = z_data_list[row][col]
	for row in range(len(z_data_list)):
		z_data_nor_list[row][col] = float(z_data_list[row][col]) / float(max+1)	

model = Sequential()

model.add(Dense(input_dim = 106, kernel_initializer = 'normal', output_dim = 1024))
model.add(Activation('relu'))
model.add(Dense(output_dim = 1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_data_nor_list, y_data_list, batch_size = 100, epochs = 2)

model.save('mymodel.h5')

result = model.predict(z_data_nor_list, batch_size = 100)

print(result)

#print(result)

with open('predict', 'w') as output:
	output.write('id,label\n')
	for row in range(len(result)):
		output.write(str(row+1))
		output.write(',')
		if(result[row] >= 0.5):
			output.write('1')
		else:
			output.write('0')
		output.write('\n')














