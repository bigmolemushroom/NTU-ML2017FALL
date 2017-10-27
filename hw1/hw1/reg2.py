import csv
import numpy as np
from math import sqrt

##### Debugging controller ========================================================================
_DISPLAY = 1

##### Number of iterations ========================================================================
ITERATION = 0

##### Parameters ==================================================================================
# Null
w_list = [0 for x in range(19)]

# Error : 6.216822314453341
w_list = [4.9382369034360485, -0.011116990110410344, -0.066895798774002396, 0.27444159644018795, -0.29018036653656576, -0.084763429140030627, 0.56953239608097828, -0.50962580043153749, -0.01249233333088217, 0.8974592033290153, -0.0005271191675845202, 0.00070768266702485782, -0.00025533393057332494, -4.2679479323243771e-05, 0.0005924683605760887, 9.2795185763396712e-05, -0.0016374834306298292, 0.00050121262104775458, 0.0021679125604778656]

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























