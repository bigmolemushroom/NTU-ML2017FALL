import numpy as np
import pandas as pd

ARGUMENTS 	= 106	# Total number of arguments
BATCH_SIZE 	= 100	# Batch size
EPOCHS 		= 1000	# Number of epochs

X_train_dir = 'X_train'
Y_train_dir = 'Y_train'

X_test_dir 	= 'X_test'
Y_test_dir	= './predict/logistic0.csv'

w_dir = './para/logistic0_w'
b_dir = './para/logistic0_b'

# 2-D array
X_train = np.array(pd.read_csv(X_train_dir))

# 1-D array
Y_train = np.array(pd.read_csv(Y_train_dir))
Y_train = np.ravel(Y_train)

# 2-D array
X_test 	= np.array(pd.read_csv(X_test_dir))

b = np.zeros((1,))
w = np.zeros((ARGUMENTS,))

l_rate = 0.001

# For batches
split = [BATCH_SIZE * i for i in range((len(X_train) // BATCH_SIZE) + 1)] + [len(X_train)]

#
for epoch in range(EPOCHS):
	print('Epochs : ', epoch, '/', EPOCHS, end = ' | ')
	for i in range(len(split)-1):
		# Split into batches
		X = X_train[split[i] : split[i+1]]
		Y = Y_train[split[i] : split[i+1]]

		z = np.dot(X, np.transpose(w)) + b
		y = np.clip(1 / (1 + np.exp(-z)), 0.00000001, 0.99999999)

		b_grad = np.mean(-1 * (Y - y))
		w_grad = np.mean(-1 * X * (Y - y).reshape((len(Y), 1)), axis = 0)

		b = b - l_rate * b_grad
		w = w - l_rate * w_grad
	
	wrong = np.count_nonzero(Y_train - (np.around(np.clip(1 / (1 + np.exp(-(np.dot(X_train, w) + b))), 0.00000001, 0.99999999))))
	print('Accuracy : ', 1 - wrong / len(Y_train))

result = np.around(np.clip(1 / (1 + np.exp( - (np.dot(X_test, w) + b))), 0.00000001, 0.99999999))
print(result)

np.savetxt(b_dir, b, delimiter = ',')
np.savetxt(w_dir, w, delimiter = ',')

with open(Y_test_dir, 'w') as output:
	output.write('id,label\n')
	for row in range(len(result)):
		output.write(str(row+1))
		output.write(',')
		output.write(str(int(result[row])))
		output.write('\n')








