from math import exp
from math import sqrt

# Functions definitions ===========================================================================
## Sigmoid function
def sigmoid(z):
	return 1/(1+exp(-z))

# Controlling variables ===========================================================================
EPOCHS = 10					# 
DISPLAY = True				# display the result or debbuging message
DISPLAY_EPOCH_FREQ = 1		# period of displaying the progress epoch
DISPLAY_ERROR_FREQ = 1		# period of computing training error

# Variables declarations ==========================================================================
x_data_list = []			# attributes of the the training data
y_data_list = []			# label of training data
w_list = []					# parameters including the bias in the end
w_grad_list = []			# gradients of the parameters
w_grad_rms_list = []		# RMS of gradients
w_lr_list = []				# learning rates

accuracy = 0
min_accuracy = 0

# Read in x_data_list =============================================================================
with open('X_train', 'r') as file:
	if(DISPLAY == 1):
		print('Reading X_train ...')
	x_data_string_list = file.read().split('\n')
	tmp_list = []
	del x_data_string_list[0]
	del x_data_string_list[-1]
	for row in range(len(x_data_string_list)):
		tmp_list.append(x_data_string_list[row].split(','))
		x_data_list.append([float(tmp_list[row][x]) for x in range(len(tmp_list[row]))] + [1.0])

# Read in y_data_list =============================================================================
with open('Y_train', 'r') as file:
	if(DISPLAY == 1):
		print('Reading Y_train ...')
	y_data_list = file.read().split('\n')
	del y_data_list[0]					# delete the title row
	del y_data_list[-1]					# delete the extra empty row
	y_data_list = [float(y_data_list[x]) for x in range(len(y_data_list))]

# Initialize w_list ===============================================================================
## Null 
w_list = [0 for i in range(len(x_data_list[0]))]
## Accuracy : 0.7639814502011609
w_list = [-6.522546492837652e-06, -6.3636997191059936e-06, -3.7262452792454244e-06, 2.9577601564279515e-05, 3.805866095821372e-05, -6.879924676041385e-06, 2.84199325885462e-06, -4.618020185211203e-06, -1.8810376195349636e-05, -1.0672711534445459e-05, 3.698712220647614e-05, -6.839442912570401e-06, -7.247036517919813e-06, -2.08314604432162e-05, -1.7491202809506e-05, -1.843467690541861e-05, -1.8889353763004498e-05, -1.83439799859082e-05, -1.7937881437742624e-05, -1.748565194971143e-05, -1.9000463892225266e-05, -1.8635522507593433e-05, -8.19922474142631e-06, -8.107325849720598e-06, 5.317754153673647e-06, 3.3712716399920396e-05, -1.4634821787272591e-05, 3.731156260515464e-05, -1.8988368868553203e-05, 3.3429815192300187e-05, -1.2995968912869994e-05, -1.7546525251666713e-05, 6.834681660374695e-06, 7.733027725177711e-06, -1.8127469165145164e-05, -1.9152602046213222e-05, -1.8086345474920145e-05, -1.86412131136703e-05, -1.576575919225055e-05, -1.5003308465912337e-05, -1.0155543327203903e-05, 1.024824319007993e-05, -1.864691802052835e-05, -1.8220241372700376e-05, -1.5801378019381107e-05, -1.9559934403530174e-05, -1.991727698287817e-05, 7.737721360893684e-06, -1.175923486407243e-06, -6.808577160962224e-06, -3.678307955503981e-06, -1.2226134078719808e-05, -1.7497638020880064e-05, 7.862461865343533e-06, -1.743837392348055e-05, -1.8894196302523355e-05, -2.001201647736286e-05, -1.878728390349583e-05, 9.676955916476851e-06, -2.1313139942846923e-05, -9.187249713015362e-06, -1.4694760992954456e-05, -1.714036259907984e-05, -8.18112869874085e-06, 1.942852091235309e-06, -3.018981505334933e-06, -7.27304148205653e-06, -1.8198118948161716e-05, -3.223568064310184e-06, -1.9084840927869923e-05, -1.4186743394908612e-05, -1.589755929747209e-05, -1.7908458255733557e-06, 5.518462181347203e-06, -2.2863436980260123e-06, -8.594317754291134e-06, -1.6749716854654665e-05, -1.592739029872632e-05, -2.7641733589218256e-05, -1.544902444371031e-05, -1.8421829148161111e-06, -9.301128065179419e-06, 3.2930813655966516e-06, 5.593210168124442e-06, -1.3885922696424596e-05, -3.6780059957270465e-07, -1.4395725406902873e-05, 2.5829156281854183e-06, -1.562930011028418e-05, -1.553010732691327e-05, -1.551886403904812e-05, -1.9524984915097814e-05, -1.5506690543101708e-05, -5.023736997079869e-06, -1.1427699963730455e-05, -1.9345472535691367e-05, -1.5687978289917378e-05, -9.101745915900235e-06, -1.3668029187815764e-05, 3.653648654985338e-06, -1.3849663174459635e-05, -1.631804320784798e-05, -9.051630344389377e-06, -1.9186199181695692e-05, 3.40648432310181e-06, -8.042042361325307e-06, -9.266497305212173e-06]

# Initialize w_grad_rms_list ======================================================================
w_grad_rms_list = [0 for i in range(len(x_data_list[0]))]

# Initialize w_lr_list ============================================================================
w_lr_list = [0.00001 for i in range(len(x_data_list[0]))]

# Training ========================================================================================
for epoch in range(EPOCHS):
	if(DISPLAY == True and epoch % DISPLAY_EPOCH_FREQ == 0):		# DISPLAY message
		print('Epochs : ', epoch, ' / ', EPOCHS)
	w_grad_list = [0 for i in range(len(x_data_list[0]))]
	for row in range(len(x_data_list)):
		xsum = 0
		for x in range(len(x_data_list[row])):
			xsum += w_list[x] * x_data_list[row][x]
		for x in range(len(x_data_list[row])):
			w_grad_list[x] -= x_data_list[row][x] * (y_data_list[row] - sigmoid(xsum))
	for x in range(len(w_list)):
		w_grad_rms_list[x] += sqrt(((w_grad_rms_list[x])**2 + (w_grad_list[x])**2) / 2).real
		w_list[x] -= w_lr_list[x] * w_grad_list[x] / w_grad_rms_list[x]
	# Error
	correct = 0
	for row in range(len(x_data_list)):
		xsum = 0
		for x in range(len(x_data_list[row])):
			xsum += w_list[x] * x_data_list[row][x]
		if(xsum < 0.5 and y_data_list[row] == 0):
			correct += 1
		elif(xsum >= 0.5 and y_data_list[row] == 1):
			correct += 1
	accuracy = correct / len(y_data_list)
	if(epoch == 0 or accuracy > min_accuracy):
		min_accuracy = accuracy
		print('w : ', w_list)
	print('Accuracy : ', accuracy)




















