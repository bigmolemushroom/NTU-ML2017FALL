import sys
import csv
import numpy as np
from math import sqrt

##### Debugging controller ========================================================================
_DISPLAY = 1

##### Number of iterations ========================================================================
ITERATION = 1000

##### Parameters ==================================================================================
# Null
w_list = [0 for x in range(19)]

# Error : 6.105844173441521
# w_list = [4.711194709090968, -0.017197019590548632, -0.065565469068645715, 0.28432684805901609, -0.29353689542880007, -0.10953749348433826, 0.60313531502714202, -0.51615385669989011, -0.072242438746427148, 0.96503775019328675, -0.00036678797040230591, 0.00044836783234715401, -0.00019560426024032643, 7.8315877997088754e-05, 0.00058499165497524178, -7.8912668349273663e-05, -0.0015393297628088062, 0.00082757726074056859, 0.0016936383932703938]

# Gradients
w_grad = [0 for x in range(19)]

# Root-sum-square of gradients over time
w_grad_rms = [0 for x in range(19)]

# Regulation
lamda = 0

# Learning rates
w_lr = [0.0000000000000001 for x in range(19)]
##### Variables ===================================================================================
data = [[] for month in range(12)]
x_list = []
y_list = []
err = 0
err_ss = 0
min_err = 0
w_best = []

##### Read in the training file ===================================================================
with open(sys.argv[1], encoding='mac_roman', newline = '') as csv_file:
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
for row in [len(x_list)-1-i for i in range(len(x_list))]:
	for hour in range(9):
		if(x_list[row][hour+1] < 0):
			del x_list[row]
			del y_list[row]
			break
		elif(y_list[row][0] < 8):
			del x_list[row]
			del y_list[row]
			break
for row in range(len(x_list)):
	for hour in range(9):
		x_list[row].append(pow(x_list[row][hour+1], 2))
x = np.array(x_list)
y = np.array(y_list)
xt = x.transpose()
w = np.dot(np.dot(np.linalg.inv(np.dot(xt, x)), xt), y)

w_list = [w[x][0] for x in range(19)]
print(w_list)
##### Training ====================================================================================
for it in range(ITERATION):
	if(_DISPLAY == 1 and it % 1 == 0):
		print("Iteration : ", it, " / ", ITERATION)
	w_grad = [0 for x in range(19)]
	for row in range(len(x_list)):
		xsum = 0
		for x in range(19):
			xsum += w_list[x]*x_list[row][x]
		w_grad = [w_grad[x] - 2 * x_list[row][x] * (y_list[row][0] - xsum) + 2 * lamda * w_list[x] for x in range(19)]
	w_grad_rms = [sqrt(pow(w_grad_rms[x], 2) + pow(w_grad[x], 2)).real for x in range(19)]
	w_list = [w_list[x] - w_lr[x] * w_grad[x] / w_grad_rms[x] for x in range(19)]

	if(_DISPLAY == 1 and it % 1 ==0):
		err_ss = 0
		for row in range(len(x_list)):
			xsum = 0
			for x in range(19):
				xsum += w_list[x]*x_list[row][x]
			err_ss += pow(y_list[row][0] - xsum, 2)
		err = sqrt(err_ss/len(x_list)).real
		if(it == 0 or err < min_err):
			min_err = err
			w_best = w_list
			print("w : ", w_list)
		print("Error : ", err)























