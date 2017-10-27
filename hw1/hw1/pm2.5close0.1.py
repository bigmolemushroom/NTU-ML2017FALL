import csv
import numpy as np
from math import sqrt

##### Debugging controller ========================================================================
_DISPLAY = 1

##### Number of iterations ========================================================================
ITERATION = 10000

##### Parameters ==================================================================================
# Null
w_list = [0 for x in range(10)]

# Error : 6.123022297751715
w_list = [1.7331687434002196, -0.033113688595748453, -0.025846973223709633, 0.21796228381646443, -0.23949949632449496, -0.054407119250515203, 0.53064384546852439, -0.57178854179297312, 0.00072210833989981332, 1.0943481695905037]

# Gradients
w_grad = [0 for x in range(10)]

# Root-sum-square of gradients over time
w_grad_rms = [0 for x in range(10)]

# Regulation
lamda = 0.1

# Learning rates
w_lr = [0.0001] + [0.0001 for x in range(9)]
##### Variables ===================================================================================
data = [[] for month in range(12)]
x_list = []
y_list = []
err = 0
err_ss = 0
min_err = 0
w_best = []

##### Read in the training file ===================================================================
with open('train.csv', encoding='mac_roman', newline = '') as csv_file:
	# Read the file and delete the first line
	csv_obj = csv.reader(csv_file)
	all_data = list(csv_obj)
	del all_data[0]
	# Setting data (18 X 5760 array)
	for month in range(12):
		for day in range(20):
			for hour in range(24):
				data[month].append(float(all_data[18*(20*month+day)+9][hour+3]))
for month in range(12):
	for index in range(471):
		x_list.append([1] + [data[month][index+x] for x in range(9)])
		y_list.append([data[month][index+9]])

x = np.array(x_list)
y = np.array(y_list)
xt = x.transpose()
w = np.dot(np.dot(np.linalg.inv(np.dot(xt, x)), xt), y)

w_list = [w[x][0] for x in range(10)]
print(w_list)
##### Train with regulation =======================================================================
for it in range(ITERATION):
	if(_DISPLAY == 1 and it % 10 == 0):
		print("Iteration : ", it, " / ", ITERATION)
	w_grad = [0 for x in range(10)]
	for row in range(len(x_list)):
		xsum = 0
		for x in range(10):
			xsum += w_list[x] * x_list[row][x]
		w_grad = [w_grad[x] - 2 * x_list[row][x] * (y_list[row][0] - xsum) + 2 * lamda * w_list[x] for x in range(10)]
	w_grad_rms = [sqrt(pow(w_grad_rms[x], 2) + pow(w_grad[x], 2)).real for x in range(10)]
	w_list = [w_list[x] - w_lr[x] * w_grad[x] / w_grad_rms[x] for x in range(10)]

	if(_DISPLAY == 1 and it % 10 == 0):
		err_ss = 0
		for row in range(len(x_list)):
			xsum = 0
			for x in range(10):
				xsum += w_list[x] * x_list[row][x]
			err_ss += pow(y_list[row][0] - xsum, 2)
		err = sqrt(err_ss/len(x_list)).real
		if(it == 0 or err < min_err):
			min_err = err
			w_best = w_list
			print("w : ", w_list)
		print("Error : ", err)

err_ss = 0
for row in range(len(x_list)):
	xsum = 0
	for x in range(10):
		xsum += w_list[x] * x_list[row][x]
	err_ss += pow(y_list[row][0] - xsum, 2)
err = sqrt(err_ss/row).real
print('Error : ', err)























