import numpy as np
import pandas as pd

ARGUMENTS 	= 106	# Total number of arguments
BATCH_SIZE 	= 100	# Batch size
EPOCHS 		= 100	# Number of epochs

X_train_dir = 'X_train'
Y_train_dir = 'Y_train'

X_test_dir 	= 'X_test'
Y_test_dir	= './predict/generative0.csv'

w_dir = './para/generative0_w'
b_dir = './para/generative0_b'


# 2-D array
X_train = np.array(pd.read_csv(X_train_dir))

# 1-D array
Y_train = np.array(pd.read_csv(Y_train_dir))
Y_train = np.ravel(Y_train)

# 2-D array
X_test 	= np.array(pd.read_csv(X_test_dir))

cnt1 = 0
cnt2 = 0

mu1 = np.zeros((106,))
mu2 = np.zeros((106,))

for i in range(len(X_train)):
	if(Y_train[i] == 1):
		mu1 += X_train[i]
		cnt1 += 1
	else:
		mu2 += X_train[i]
		cnt2 += 1

mu1 /= cnt1
mu2 /= cnt2

sigma1 = np.zeros((106, 106))
sigma2 = np.zeros((106, 106))

for i in range(len(X_train)):
	if(Y_train[i] == 1):
		sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
	else:
		sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])

sigma1 /= cnt1
sigma2 /= cnt2

shared_sigma = (float(cnt1) / len(X_train)) * sigma1 + (float(cnt2) / len(X_train)) * sigma2

sigma_inverse = np.linalg.inv(shared_sigma)
w = np.dot( (mu1-mu2), sigma_inverse)
x = X_train.T
b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(cnt1)/cnt2)
a = np.dot(w, x) + b
y = np.clip(1 / (1 + np.exp(-a)), 0.00000001, 0.99999999)
result = np.around(y)
wrong = np.count_nonzero(Y_train - result)
accuracy = 1 - wrong / len(Y_train)
print('Train accuracy : ', accuracy)

np.savetxt(b_dir, b, delimiter = ',')
np.savetxt(w_dir, w, delimiter = ',')

with open(Y_test_dir, 'w') as output:
	sigma_inverse = np.linalg.inv(shared_sigma)
	w = np.dot( (mu1-mu2), sigma_inverse)
	x = X_test.T
	b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(cnt1)/cnt2)
	a = np.dot(w, x) + b
	y = np.clip(1 / (1 + np.exp(-a)), 0.00000001, 0.99999999)
	result = np.around(y)
	output.write('id,label\n')
	for i in range(len(X_test)):
		output.write(str(i+1))
		output.write(',')
		output.write(str(int(result[i])))
		output.write('\n')











