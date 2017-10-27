import csv
from cmath import sqrt

arg_enable = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

w = [4.711194709090968, -0.017197019590548732, -0.065565469068645812, 0.28432684805901598, -0.29353689542879996, -0.10953749348433817, 0.60313531502714213, -0.51615385669989, -0.072242438746427051, 0.96503775019328686, -0.00036678797040240593, 0.00044836783234705399, -0.00019560426024042642, 7.8315877997188759e-05, 0.00058499165497514181, -7.8912668349373667e-05, -0.0015393297628089062, 0.00082757726074066855, 0.0016936383932704938]

#b = 1.6991554445670143
#w = [[0 for x in range(9)] for arg in range(18)]
#w[9] = [-0.04151721428417211, -0.022806293811867145, 0.21638018030288617, -0.23031805557403606, -0.05444472694607248, 0.5334947052941601, -0.564716580410561, -0.004373638781132588, 1.0927220550877947]

data = []

with open('test.csv', encoding='mac_roman', newline = '') as csv_file:
	csv_obj = csv.reader(csv_file)
	all_data = list(csv_obj)

	for index in range(240):
		for arg in range(18):
			arg_data = []
			if(arg == 10):
				rainfall = []
				del all_data[18*index+10][0:2]
				for x in range(9):
					if(all_data[18*index+10][x] == 'NR'):
						rainfall.append(0)
					else:
						rainfall.append(1)
				arg_data += rainfall
			else:
				del all_data[18*index+arg][0:2]
				arg_data += all_data[18*index+arg]
			data.append(arg_data)
	
	with open('./aa/sampleSubmission.csv', 'w') as output:
		output.write('id,value\n')
		for n in range(240):
			output.write('id_')
			output.write(str(n))
			output.write(',')
			pre = w[0]
			for x in range(9):
				pre += w[x+1]*float(data[n*18+9][x]) + w[x+10]*pow(float(data[n*18+9][x]), 2)
			print(n, ' : ', pre)
			output.write(str(pre))
			output.write('\n')
