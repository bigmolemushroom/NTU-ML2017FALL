import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.models import load_model

x_data_list = np.array(pd.read_csv('X_train'))
y_data_list = np.array(pd.read_csv('Y_train'))
z_data_list = np.array(pd.read_csv('X_test'))

x_max = np.amax(x_data_list, axis = 0)

x_data_list = x_data_list / x_max
z_data_list = z_data_list / x_max

model = Sequential()

model.add(Dense(input_dim = 106, kernel_initializer = 'normal', output_dim = 1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(kernel_initializer = 'normal', output_dim = 1024, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim = 1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_data_list, y_data_list, batch_size = 100, epochs = 25)

model.save('best3.h5')

result = model.predict(z_data_list, batch_size = 100)

print(result)

with open('best3', 'w') as output:
	output.write('id,label\n')
	for row in range(len(result)):
		output.write(str(row+1))
		output.write(',')
		if(result[row] >= 0.5):
			output.write('1')
		else:
			output.write('0')
		output.write('\n')














