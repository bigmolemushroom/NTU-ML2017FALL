import csv
import numpy as np
from math import sqrt

##### Debugging controller ========================================================================
_DISPLAY = 1

##### Number of iterations ========================================================================
ITERATION = 0

##### Parameters ==================================================================================
# Null
#w_list = [0 for x in range(91)]

# Error : 5.805573316783852
w_list = [-0.034298775508178614, -0.11007344221924809, 0.077081630527199407, -0.064152923476128332, -0.26881479099744277, 0.35226209760055066, -0.13023064573914489, 0.22454290012011313, -2.1501771667930161, 0.68071013403976099, 4.378901844891173, 0.13595466596883221, -0.046959079020791106, -0.46137023386636467, 0.31467095584294236, 1.1849098005147976, 3.3515409424496703, -3.4027215354140319, -0.52396257943855851, 1.9939916536353381, 0.50860354746080549, -0.21902498009313448, -0.16548772299892084, 0.02675780485037893, -0.28077293345865506, -0.24224778726108331, -0.32316797791950752, -0.056247657560350828, -0.093687698065404945, -0.48496239538293306, 0.081955109120999126, 0.25335824665577666, 0.079131346989903872, 0.020979853672458249, 0.38306019623667531, 0.21120185701528515, -0.018102663659164263, -0.023459156586066128, -0.029334889479283081, -0.011396345939365775, 0.10289960812014362, 0.0053777051478386233, -0.019300741510421163, 0.019840783858485693, -0.0003017631009256394, 0.042508196189338307, -0.062487260046972706, 0.38731559197959653, -0.45254022679142758, 0.0053891999598861123, 0.93753000162604661, -0.051922633662949962, 0.05188845268784717, 0.014822741404279416, -0.034709356990836153, -0.085617824978248019, -0.067916393348317672, 0.085173468228007768, -0.12034222295947908, 0.016373498505913804, 0.051960217343916516, -0.033803773163427031, 0.055476442749781474, -0.10114477650444065, 0.13998923584065903, 0.12988330126424585, -0.99702188126686409, 1.3749878522155141, 1.3163984340342409, -2.2321494251626639, -0.72522206758437946, 0.00054459820561714774, 0.0020450306109281702, -0.002775100905905695, 0.0012980451521617169, 0.00057727703396827032, 0.00044235692729163072, 0.00019478379141918412, -2.2725042653997871e-05, -0.002605678959734254, 7.8119437340212414e-05, -0.057645541303317893, -0.091356527178507119, -0.038631587022400049, -0.060081982044014978, -0.062144273027794461, -0.14437877593130286, 0.30024273907116533, -0.032866362577290628, -0.25073201572445331, 0.13432329231062784]

# Gradients
w_grad = [0 for x in range(91)]

# Root-sum-square of gradients over time
w_grad_rms = [0 for x in range(91)]

# Regulation
lamda = 0

# Learning rates
w_lr = [0.001] + [0.001 for x in range(90)]
##### Variables ===================================================================================
data = [[[] for arg in range(18)] for month in range(12)]
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
				for arg in range(18):
					if(all_data[18*(20*month+day)+arg][hour+3] == 'NR'):
						data[month][arg].append(0)
					else:
						data[month][arg].append(float(all_data[18*(20*month+day)+arg][hour+3]))
for month in range(12):
	for index in range(475):
		tmplist = [1]
		for arg in range(18):
			tmplist += [data[month][arg][index+x] for x in range(5)]
		x_list.append(tmplist)
		y_list.append([data[month][9][index+5]])

x = np.array(x_list)
y = np.array(y_list)
xt = x.transpose()
w = np.dot(np.dot(np.linalg.inv(np.dot(xt, x)), xt), y)

w_list = [w[x][0] for x in range(91)]
print(w_list)
##### Train with regulation =======================================================================
for it in range(ITERATION):
	if(_DISPLAY == 1 and it % 10 == 0):
		print("Iteration : ", it, " / ", ITERATION)
	w_grad = [0 for x in range(91)]
	for row in range(len(x_list)):
		xsum = 0
		for x in range(91):
			xsum += w_list[x] * x_list[row][x]
		w_grad = [w_grad[x] - 2 * x_list[row][x] * (y_list[row][0] - xsum) + 2 * lamda * w_list[x] for x in range(91)]
	w_grad_rms = [sqrt(pow(w_grad_rms[x], 2) + pow(w_grad[x], 2)).real for x in range(91)]
	w_list = [w_list[x] - w_lr[x] * w_grad[x] / w_grad_rms[x] for x in range(91)]

	if(_DISPLAY == 1 and it % 10 == 0):
		err_ss = 0
		for row in range(len(x_list)):
			xsum = 0
			for x in range(91):
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
	for x in range(91):
		xsum += w_list[x] * x_list[row][x]
	err_ss += pow(y_list[row][0] - xsum, 2)
err = sqrt(err_ss/row).real
print('Error : ', err)























